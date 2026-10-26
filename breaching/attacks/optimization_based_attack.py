"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

from omegaconf import open_dict

log = logging.getLogger(__name__)


class OptimizationBasedAttacker(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu")), **kwargs):
        super().__init__(model, loss_fn, cfg_attack, setup)
        self.parallel_trials = False
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        log.info(f"Using objective function: {self.cfg.objective}")
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
            if hasattr(self.objective, 'pop_index'):
                log.info("The objective function is set to optimize over positional embeddings separately from the rest of the parameters.")
                # log.warning(f"kwargs is {kwargs}")
                cfg_peft = kwargs.get("cfg_peft", None)
                # log.warning(f"cfg_peft is {cfg_peft}")
                if cfg_peft is not None and cfg_peft.enabled:
                    self.objective.pop_index = 2 if cfg_peft.type == "vpt" else 0
                    log.info(f"Based on the peft type {cfg_peft.type}, the objective will pop the gradients at index {self.objective.pop_index} for the posembed optimization.")
                else:
                    log.warning("Defaulting to pop_index 1, which may not be correct for all model architectures (CNN).")
            else:
                log.info("The objective function optimizes positional embeddings together with the rest of the parameters by default.")

        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                # Add a separate list for parallel operations like Group Regularization, Group Lazy Regularization
                if hasattr(self.cfg.regularization[key], 'parallel'):
                    if self.cfg.regularization[key].parallel:
                        self.parallel_trials = True
                        if not hasattr(self, 'parallel_regularizers'):
                            self.parallel_regularizers = []
                        self.parallel_regularizers.append(regularizer_lookup[key](self.setup, **self.cfg.regularization[key]))
                        # Should not include these regularizers in serial operations / regularizers
                        continue
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        except AttributeError:
            pass  # No regularizers selected.

        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.

        for regularizer in self.regularizers:
            # print(regularizer.__class__.__name__)
            if regularizer.__class__.__name__ == "ICAFeatureRegularization":
                with open_dict(self.cfg.impl):
                    self.cfg.impl.ica_feature_reg = True
                log.info("ICA regularization selected. Make sure that the target model architecture and the corresponding layer for feature extraction are compatible with this regularizer!")
        # print(self.cfg.impl)

    def __repr__(self):
        n = "\n"
        parallel_regs = getattr(self, "parallel_regularizers", [])
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
        Hyperparameter Template: {self.cfg.type}

        Objective: {repr(self.objective)}
        Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
        Parallel Regularizers: {(n + ' '*27).join([repr(r) for r in parallel_regs]) if parallel_regs else 'None'}
        Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

        Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        log.info(f"Starting reconstruction with {self.cfg.restarts.num_trials} trial(s). Parallel trials: {self.parallel_trials}")
        log.info(f"Number of models used for reconstruction: {len(rec_models)}")
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            if not self.parallel_trials:
                log.info(f"Running serial trials...")
                for trial in range(self.cfg.restarts.num_trials):
                    candidate_solutions += [self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)]
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
            else:
                log.info(f"Running parallel trials...")
                candidate_solutions = self._run_parallel_trials(rec_models, 
                                                                shared_data, 
                                                                labels, stats, 
                                                                initial_data, 
                                                                dryrun, 
                                                                self.cfg.restarts.num_trials)
                for trial in range(self.cfg.restarts.num_trials):
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats, [dict(data=candidate, labels=labels) for candidate in candidate_solutions]
    
    def _run_parallel_trials(self, rec_model, shared_data, labels, stats, initial_data, dryrun, num_trials):
        f"""Run a {num_trials} reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        log.info(f"Data Shape: {self.data_shape}")
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        candidates = [candidate.detach().clone().requires_grad_(True) for _ in range(num_trials)]
        log.info(f"Candidate Shape: {candidate.shape}")
        if initial_data is not None:
            for cand in candidates:
                cand.data = initial_data.data.clone().to(**self.setup)
        
        best_candidates = [candidate.detach().clone() for candidate in candidates]
        minimal_values_so_far = torch.as_tensor([float("inf")] * num_trials, **self.setup)

        # Initialize optimizers
        optimizers, schedulers = [], []
        use_single_optimizer = self.cfg.optim.optimizer.lower() == "l-bfgs"
        optimizer = None
        scheduler = None
        if use_single_optimizer:
            optimizer, scheduler = self._init_optimizer(candidates)
        else:
            for candidate in candidates:
                opt, sched = self._init_optimizer([candidate])
                optimizers.append(opt)
                schedulers.append(sched)

        current_wallclock = time.time()
        parallel_regularizers = getattr(self, "parallel_regularizers", [])
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                if use_single_optimizer:
                    closure = self._compute_combined_objective(
                        candidates,
                        labels,
                        rec_model,
                        optimizer,
                        optimizers,
                        shared_data,
                        iteration,
                        parallel_regularizers,
                        use_single_optimizer,
                    )
                    optimizer.step(closure)
                    scheduler.step()
                else:
                    closure = self._compute_combined_objective(
                        candidates,
                        labels,
                        rec_model,
                        optimizer,
                        optimizers,
                        shared_data,
                        iteration,
                        parallel_regularizers,
                        use_single_optimizer,
                    )
                    _ = closure()
                    for opt, sched in zip(optimizers, schedulers):
                        opt.step()
                        sched.step()

                objective_values = self.current_objective_values
                task_losses = self.current_task_losses
                
                with torch.no_grad():
                    # Project into image space
                    for idx, (candidate, objective_value) in enumerate(zip(candidates, objective_values)):
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        if objective_value < minimal_values_so_far[idx]:
                            minimal_values_so_far[idx] = objective_value.detach()
                            best_candidates[idx] = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    for idx, (objective_value, task_loss) in enumerate(zip(objective_values, task_losses)):
                        log.info(
                            f"| Trial: {idx} | It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                    current_wallclock = timestamp

                    if not torch.isfinite(objective_value):
                        log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                        break
                for trial in range(self.cfg.restarts.num_trials):
                    stats[f"Trial_{trial}_Val"].append(objective_values[trial].item())

                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        return best_candidates

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for idx, (model, data) in enumerate(zip(rec_model, shared_data)):
                # log.info(f"Computing objective for model {idx}...")
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            # log.info(f"Objective Loss: {objective.item():2.4f}")
            # log.info(f"Number of regularizers: {len(self.regularizers)}")
            for regularizer in self.regularizers:
                if hasattr(regularizer, "x_list"):
                    regularizer.x_list = getattr(self, "candidate_solutions", None)
                if hasattr(regularizer, "aligner") and regularizer.aligner is None:
                    regularizer.aligner = self.aligner
                try:
                    temp_loss = regularizer(candidate_augmented)
                except Exception as e:
                    log.error(f"Error in regularizer {repr(regularizer)}: {e}")
                    raise e
                # log.info(f"Regularizer Loss: {temp_loss.item():2.4f}")
                total_objective += temp_loss

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _compute_combined_objective(
        self,
        candidates,
        labels,
        rec_model,
        optimizer,
        optimizers,
        shared_data,
        iteration,
        parallel_regularizers,
        use_single_optimizer,
    ):
        def apply_grad_transforms(candidate, opt):
            if self.cfg.optim.langevin_noise > 0:
                step_size = opt.param_groups[0]["lr"]
                noise_map = torch.randn_like(candidate.grad)
                candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
            if self.cfg.optim.grad_clip is not None:
                grad_norm = candidate.grad.norm()
                if grad_norm > self.cfg.optim.grad_clip:
                    candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
            if self.cfg.optim.signed is not None:
                if self.cfg.optim.signed == "soft":
                    scaling_factor = (
                        1 - iteration / self.cfg.optim.max_iterations
                    )  # just a simple linear rule for now
                    candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                elif self.cfg.optim.signed == "hard":
                    candidate.grad.sign_()

        def closure():
            if use_single_optimizer:
                optimizer.zero_grad()
            else:
                for opt in optimizers:
                    opt.zero_grad()

            objective_values = []
            task_losses = []

            for candidate in candidates:
                if self.cfg.differentiable_augmentations:
                    candidate_augmented = self.augmentations(candidate)
                else:
                    candidate_augmented = candidate
                    candidate_augmented.data = self.augmentations(candidate.data)

                total_objective = 0
                total_task_loss = 0
                for idx, (model, data) in enumerate(zip(rec_model, shared_data)):
                    # log.info(f"Computing objective for model {idx}...")
                    objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                    total_objective += objective
                    total_task_loss += task_loss
                for regularizer in self.regularizers:
                    try:
                        total_objective += regularizer(candidate_augmented)
                    except Exception as e:
                        log.error(f"Error in regularizer {repr(regularizer)}: {e}")
                        raise e

                objective_values.append(total_objective)
                task_losses.append(total_task_loss)

            for regularizer in parallel_regularizers:
                try:
                    reg_loss = regularizer(candidates)
                    for idx, reg in enumerate(reg_loss):
                        objective_values[idx] = objective_values[idx] + reg
                except Exception as e:
                    log.error(f"Error in parallel regularizer {repr(regularizer)}: {e}")
                    raise e

            total_objective = sum(objective_values)
            if total_objective.requires_grad:
                total_objective.backward(inputs=candidates, create_graph=False)
            with torch.no_grad():
                if use_single_optimizer:
                    for candidate in candidates:
                        apply_grad_transforms(candidate, optimizer)
                else:
                    for candidate, opt in zip(candidates, optimizers):
                        apply_grad_transforms(candidate, opt)

            self.current_objective_values = objective_values
            self.current_task_losses = task_losses
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)
