""" Implements the CGIR attacks on ViT models.
This method uses L2 loss optimization for the parameter gradients,
And the positional embedding gradients are also recovered using cosine similarity loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from pathlib import Path

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

from .auxiliaries.BigGAN.model import (
    BigGAN,
    Generator,
    biggan_model_name_for_output_dim,
    resolve_pretrained_biggan_weights,
)
from .auxiliaries.BigGAN.config import BigGANConfig, BigGAN32, BigGAN128, BigGAN256, BigGAN512
from .auxiliaries.BigGAN.utils import truncated_noise_sample, save_as_images, one_hot_from_names, one_hot_from_int

from .auxiliaries.stylegan_xl import legacy, dnnlib
from .auxiliaries.stylegan_xl.torch_utils import misc
from .auxiliaries.stylegan2_ada import legacy as stylegan2_ada_legacy, dnnlib as stylegan2_ada_dnnlib
from .auxiliaries.stylegan2_ada.torch_utils import misc as stylegan2_ada_misc

from .auxiliaries.dcgan.dcgan import Generator as DCGANGenerator

import logging

log = logging.getLogger(__name__)

def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ['state_dict', 'model', 'netG', 'generator', 'G_ema', 'G', 'g_ema']:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state

def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}

def _load_state_dict_forgiving(model, state_dict):
    state_dict = _strip_module_prefix(state_dict)
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append((k, getattr(v, 'shape', None), model_sd.get(k).shape if k in model_sd else None))
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        log.warning("Skipped %d generator weights due to shape/key mismatch.", len(skipped))
    if missing:
        log.warning("Generator missing %d keys after load.", len(missing))
    if unexpected:
        log.warning("Generator got %d unexpected keys after load.", len(unexpected))

def build_stylegan_xl_generator(
    device,
    network_pkl=None,
    class_name='training.networks_stylegan2.Generator',
    init_kwargs=None,
    common_kwargs=None,
):
    """Construct a StyleGAN-XL generator and optionally load generator-only weights from a pickle."""
    common_kwargs = common_kwargs or {}
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            data = legacy.load_network_pkl(f)
        src_g = data['G_ema']
        log.info(f"StyleGAN-XL init_kwargs: {src_g.init_kwargs}")
        # Prefer the generator defined in the pickle to avoid class/kwargs mismatches.
        if init_kwargs is None:
            return src_g.to(device).eval()
        # If caller insists on reconstruction, use the pickle's class_name unless overridden.
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', src_g.init_kwargs.get('class_name', class_name))
        G = dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()
        misc.copy_params_and_buffers(src_g, G, require_all=False)
        return G

    if init_kwargs is None:
        init_kwargs = dict(class_name=class_name, **common_kwargs)
    else:
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', class_name)
        init_kwargs.update(common_kwargs)
    return dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()

def build_stylegan2_ada_generator(
    device,
    network_pkl=None,
    class_name='training.networks.Generator',
    init_kwargs=None,
    common_kwargs=None,
):
    """Construct a StyleGAN2-ADA generator and optionally load generator-only weights from a pickle."""
    common_kwargs = common_kwargs or {}
    if network_pkl is not None:
        with stylegan2_ada_dnnlib.util.open_url(network_pkl) as f:
            data = stylegan2_ada_legacy.load_network_pkl(f)
        src_g = data['G_ema']
        if init_kwargs is None:
            return src_g.to(device).eval()
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', src_g.init_kwargs.get('class_name', class_name))
        G = stylegan2_ada_dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()
        stylegan2_ada_misc.copy_params_and_buffers(src_g, G, require_all=False)
        return G

    if init_kwargs is None:
        init_kwargs = dict(class_name=class_name, **common_kwargs)
    else:
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', class_name)
        init_kwargs.update(common_kwargs)
    return stylegan2_ada_dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()

class GenImagePriorAttacker(OptimizationBasedAttacker):
    """Implements an optimization-based attack that only recovers the patch information 
    by opyimization L2 loss on the the parameter gradients.
    And also optimizes the cosine similarity loss on the PosEmbed gradients
    """
    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu")), **kwargs):
        super().__init__(model, loss_fn, cfg_attack, setup, **kwargs)
        self.truncation = self.cfg.objective.truncation if hasattr(self.cfg.objective, 'truncation') else 0.4
        self._biggan_weights_path = None
        self._biggan_state_dict = None

    def _resize_dcgan_output(self, x):
        if self.cfg.generator.type != "dcgan":
            return x
        if x.ndim != 4 or len(self.data_shape) < 3:
            return x
        target_h, target_w = self.data_shape[-2], self.data_shape[-1]
        if x.shape[-2:] != (target_h, target_w):
            return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    def _make_cond_label(self, labels, device, c_dim):
        if c_dim == 0:
            return torch.zeros([self.batch_size, 0], device=device)
        label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
        return torch.tensor(label, dtype=torch.float, device=device)

    def _resolve_biggan_weights(self):
        if self._biggan_weights_path is not None:
            return self._biggan_weights_path
        if not hasattr(self.cfg, "generator"):
            return None
        use_pretrained = bool(
            getattr(self.cfg.generator, "pretrained", False) or getattr(self.cfg.generator, "network_wts", None)
        )
        if not use_pretrained:
            return None
        model_name = biggan_model_name_for_output_dim(self.bigganconfig.output_dim)
        if model_name is None:
            log.info("No pretrained BigGAN available for output_dim=%s.", self.bigganconfig.output_dim)
            return None
        if self.bigganconfig.num_classes != 1000:
            log.info(
                "Skipping pretrained BigGAN weights for num_classes=%s (expected 1000).",
                self.bigganconfig.num_classes,
            )
            return None
        requested_path = getattr(self.cfg.generator, "network_wts", None)
        weights_path = resolve_pretrained_biggan_weights(model_name, requested_path=requested_path)
        if not weights_path:
            return None
        self.cfg.generator.network_wts = weights_path
        self._biggan_weights_path = weights_path
        return weights_path

    def _canonicalize_github_raw(self, url):
        if "github.com" in url and "/blob/" in url:
            return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return url

    def _resolve_dcgan_weights(self):
        if not hasattr(self.cfg, "generator"):
            return None
        if self.cfg.generator.type != "dcgan":
            return None
        requested_path = getattr(self.cfg.generator, "network_wts", None)
        if requested_path and os.path.exists(requested_path):
            return requested_path

        default_url = (
            "https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/"
            "cifar10_dcgan/weights/netG_epoch_199.pth"
        )
        download_url = self._canonicalize_github_raw(requested_path or default_url)
        package_root = Path(__file__).resolve().parents[1]
        target_dir = package_root / "pretrained" / "DCGAN"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / os.path.basename(download_url)

        if not target_path.exists():
            log.info("Downloading DCGAN weights from %s to %s", download_url, target_path)
            torch.hub.download_url_to_file(download_url, str(target_path))

        self.cfg.generator.network_wts = str(target_path)
        return str(target_path)

    def _generate_candidate(self, netG, noise, labels):
        if self.cfg.generator.type == "biggan":
            label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
            candidate = netG(noise, label, truncation=self.truncation)
        elif self.cfg.generator.type in ("stylegan_xl", "styleganxl"):
            label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
            candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
        elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
            candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
        elif self.cfg.generator.type == "dcgan":
            candidate = netG(noise)
        else:
            raise ValueError(f"Unknown generator type {self.cfg.generator.type}.")
        return self._resize_dcgan_output(candidate)

    def _map_generator_output_to_model_space(self, candidate):
        """Map generator output from tanh space [-1, 1] to model-normalized space."""
        # print("current candidate range: [{:.4f}, {:.4f}]".format(candidate.min().item(), candidate.max().item()))
        if not hasattr(self, "dm") or not hasattr(self, "ds"):
            return candidate
        if self.dm.ndim == 0 or self.ds.ndim == 0:
            return candidate
        candidate_01 = (candidate + 1.0) * 0.5
        candidate_01 = (candidate_01 - self.dm) / self.ds
        # print("candidate in normalized range: [{:.4f}, {:.4f}]".format(candidate_01.min().item(), candidate_01.max().item()))
        # print("model normalization mean: {}, std: {}".format(self.dm, self.ds))
        return candidate_01

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        self.num_classes = server_payload[0]["metadata"]["classes"]
        log.info(f"Number of classes in the dataset is {self.num_classes}.")
        if self.cfg.generator.type == "biggan":
            if hasattr(server_payload[0]["metadata"], "name") and "CIFAR" in server_payload[0]["metadata"].name:
                self.bigganconfig = BigGAN32
            elif hasattr(server_payload[0]["metadata"], "name") and "ImageNet" in server_payload[0]["metadata"].name:
                self.bigganconfig = BigGAN256
            self.bigganconfig.num_classes = self.num_classes
        
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            if not self.parallel_trials:
                for trial in range(self.cfg.restarts.num_trials):
                    candidate_solutions += [
                        self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                    ]
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
            else:
                candidate_solutions = self._run_parallel_trials(
                    rec_models, shared_data, labels, stats, initial_data, dryrun, self.cfg.restarts.num_trials
                )
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

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
        # Initialize candidate reconstruction data
        self.batch_size = shared_data[0]["metadata"]["num_data_points"]
        # GIRG Does not require a pre-trained BigGAN
        # so we can directly use the architecture and randomly initialized weights for the attack
        if self.cfg.generator.type == "biggan":
            self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
            weights_path = self._resolve_biggan_weights()
            if weights_path:
                if self._biggan_state_dict is None:
                    self._biggan_state_dict = torch.load(weights_path, map_location="cpu")
                self.netG.load_state_dict(self._biggan_state_dict, strict=False)
                log.info(f"Loaded BigGAN weights from {weights_path}")
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.config.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type == "stylegan_xl":
            self.netG = build_stylegan_xl_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            self.netG = build_stylegan2_ada_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size,
                                            dim_z=self.netG.z_dim,
                                            truncation=self.truncation)
        elif self.cfg.generator.type == "dcgan":
            self.netG = DCGANGenerator(nz=self.cfg.generator.z_dim, ngf=self.cfg.generator.ngf, nc=self.cfg.generator.nc).to(self.setup["device"])
            weights_path = self._resolve_dcgan_weights()
            if not weights_path:
                raise ValueError("DCGAN weights are required but could not be resolved.")
            state = torch.load(weights_path, map_location=self.setup["device"])
            # state = _unwrap_state_dict(state)
            # _load_state_dict_forgiving(self.netG, state)
            self.netG.load_state_dict(state)
            # Define noise
            # self.noise = truncated_noise_sample(batch_size=self.batch_size, 
            #                                 dim_z=self.cfg.generator.z_dim, 
            #                                 truncation=self.truncation)
            self.noise = torch.randn(self.batch_size, self.cfg.generator.z_dim, 1, 1, device=self.setup["device"])

        log.info("The number of parameters in the generator is {}".format(sum(p.numel() for p in self.netG.parameters())))
        log.info("Shape of Noise is {}".format(self.noise.shape))
        log.info("### OPTIMIZING LATENT CODE ONLY FOR TRIAL {} ###".format(trial))
        # log.info(f"self.netG.embeddings.weight.shape = {self.netG.embeddings.weight.shape}")
        for p in self.netG.parameters():
            p.requires_grad = False

        if isinstance(self.noise, np.ndarray):
            self.noise = torch.tensor(self.noise, **self.setup).detach().clone().requires_grad_(True)
        else:
            self.noise = self.noise.detach().clone().requires_grad_(True)
        
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
        # Initialize optimizers
        # In GIAS, we optimzie the latent code first
        optimizer, scheduler = self._init_optimizer([self.noise])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective_latent(self.netG, 
                                                        self.noise,
                                                        labels, 
                                                        rec_model, 
                                                        optimizer, 
                                                        shared_data, # Contains both gradients and posembed gradients
                                                        iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                # log.info(f"Objective Value: {objective_value.item():.4f}, Task Loss: {task_loss.item():.4f}")
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.generator.type == "biggan":
                        label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                        candidate = self.netG(self.noise, label, truncation=self.truncation)
                    elif self.cfg.generator.type == "styleganxl":
                        label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                        label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type == "dcgan":
                        candidate = self.netG(self.noise)
                    candidate = self._resize_dcgan_output(candidate)
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
                    # log.info("Norm of Generator parameters after {} iters of optimization: {}".format(iteration, sum(p.norm().item() for p in self.netG.parameters())))

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        
        log.info("### OPTIMIZING GENERATOR PARAMS ONLY FOR TRIAL {} ###".format(trial))
        ###################################################
        ### parameter OPTIMIZATION STARTS HERE  ###########
        ###################################################
        log.info("Starting fine level optimization for trial {} with the best candidate from coarse optimization.".format(trial))
        for p in self.netG.parameters():
            p.requires_grad = True

        self.noise = self.noise.requires_grad_(False)

        with torch.no_grad():
            if self.cfg.generator.type == "biggan":
                label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                candidate = self.netG(self.noise, label, truncation=self.truncation)
            elif self.cfg.generator.type == "stylegan_xl":
                label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type == "dcgan":
                candidate = self.netG(self.noise)
            candidate = self._resize_dcgan_output(candidate)
        
        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
        # Initialize optimizers
        # Since _init_optimizer only looks into self.cfg.optim,
        # we need to manually overwrite the optimization parameters here to use the ones for parameter optimization
        prev_optim = {
            "optimizer": self.cfg.optim.optimizer,
            "step_size": self.cfg.optim.step_size,
            "max_iterations": self.cfg.optim.max_iterations,
            "callback": self.cfg.optim.callback,
        }
        self.cfg.optim.optimizer = self.cfg.optim_w.optimizer
        self.cfg.optim.step_size = self.cfg.optim_w.step_size
        self.cfg.optim.max_iterations = self.cfg.optim_w.max_iterations
        self.cfg.optim.callback = self.cfg.optim_w.callback

        optimizer, scheduler = self._init_optimizer([p for p in self.netG.parameters() if p.requires_grad])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(self.netG, 
                                                    self.noise,
                                                    labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    if self.cfg.generator.type == "biggan":
                        label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                        candidate = self.netG(self.noise, label, truncation=self.truncation)
                    elif self.cfg.generator.type == "stylegan_xl":
                        label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                        label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type == "dcgan":
                        candidate = self.netG(self.noise)
                    candidate = self._resize_dcgan_output(candidate)

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
        finally:
            self.cfg.optim.optimizer = prev_optim["optimizer"]
            self.cfg.optim.step_size = prev_optim["step_size"]
            self.cfg.optim.max_iterations = prev_optim["max_iterations"]
            self.cfg.optim.callback = prev_optim["callback"]

        self.netG = None # Free up memory

        return best_candidate.detach()

    def _run_parallel_trials(self, rec_model, shared_data, labels, stats, initial_data, dryrun, num_trials):
        """Run multiple GIAS trials in parallel for parallel regularizers."""
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        self.batch_size = shared_data[0]["metadata"]["num_data_points"]
        netGs, noises = [], []
        for _ in range(num_trials):
            if self.cfg.generator.type == "biggan":
                netG = BigGAN(self.bigganconfig).to(self.setup["device"])
                weights_path = self._resolve_biggan_weights()
                if weights_path:
                    if self._biggan_state_dict is None:
                        self._biggan_state_dict = torch.load(weights_path, map_location="cpu")
                    netG.load_state_dict(self._biggan_state_dict, strict=False)
                    log.info(f"Loaded BigGAN weights from {weights_path}")
                noise = truncated_noise_sample(
                    batch_size=self.batch_size, dim_z=netG.config.z_dim, truncation=self.truncation
                )
                noise = torch.tensor(noise, **self.setup)
            elif self.cfg.generator.type in ("stylegan_xl", "styleganxl"):
                netG = build_stylegan_xl_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
                noise = truncated_noise_sample(batch_size=self.batch_size, dim_z=netG.z_dim, truncation=self.truncation)
                noise = torch.tensor(noise, **self.setup)
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                netG = build_stylegan2_ada_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
                noise = truncated_noise_sample(batch_size=self.batch_size, dim_z=netG.z_dim, truncation=self.truncation)
                noise = torch.tensor(noise, **self.setup)
            elif self.cfg.generator.type == "dcgan":
                netG = DCGANGenerator(
                    nz=self.cfg.generator.z_dim, ngf=self.cfg.generator.ngf, nc=self.cfg.generator.nc
                ).to(self.setup["device"])
                weights_path = self._resolve_dcgan_weights()
                if not weights_path:
                    raise ValueError("DCGAN weights are required but could not be resolved.")
                state = torch.load(weights_path, map_location=self.setup["device"])
                missing, unexpected = netG.load_state_dict(state)
                # print("missing keys in generator state dict after load: {}".format(missing  if missing else "None"))
                # print("unexpected keys in generator state dict after load: {}".format(unexpected if unexpected else "None"))
                noise = torch.randn(self.batch_size, self.cfg.generator.z_dim, 1, 1, device=self.setup["device"])
            else:
                raise ValueError(f"Unknown generator type {self.cfg.generator.type}.")

            netGs.append(netG)
            noises.append(noise.detach().clone())

        for netG in netGs:
            for p in netG.parameters():
                p.requires_grad = False
        noises = [noise.requires_grad_(True) for noise in noises]

        with torch.no_grad():
            best_candidates = [self._generate_candidate(netG, noise, labels).detach().clone() for netG, noise in zip(netGs, noises)]
        minimal_values_so_far = torch.as_tensor([float("inf")] * num_trials, **self.setup)

        optimizers, schedulers = [], []
        for noise in noises:
            optimizer, scheduler = self._init_optimizer([noise])
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        parallel_regularizers = getattr(self, "parallel_regularizers", [])
        #print("parallel_regularizers: {}".format(parallel_regularizers))
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                objective_values, task_losses, candidates = [], [], []
                candidate_for_models = []
                for netG, noise in zip(netGs, noises):
                    candidate = self._generate_candidate(netG, noise, labels)
                    if self.cfg.differentiable_augmentations:
                        candidate_augmented = self.augmentations(candidate)
                    else:
                        candidate_augmented = candidate
                        candidate_augmented.data = self.augmentations(candidate.data)
                    candidate_for_model = self._map_generator_output_to_model_space(candidate_augmented)

                    total_objective = 0
                    total_task_loss = 0
                    for model, data in zip(rec_model, shared_data):
                        objective, task_loss = self.objective(model, data["gradients"], candidate_for_model, labels)
                        total_objective += objective
                        total_task_loss += task_loss
                    for regularizer in self.regularizers:
                        if regularizer.__class__.__name__ in ["AdversarialDiscriminatorRegularization"]:
                            total_objective += regularizer(candidate_augmented)
                        else:
                            total_objective += regularizer(candidate_for_model)

                    objective_values.append(total_objective)
                    task_losses.append(total_task_loss)
                    candidates.append(candidate)
                    candidate_for_models.append(candidate_for_model)

                for regularizer in parallel_regularizers:
                    reg_loss = regularizer(candidate_for_models)
                    for idx, reg in enumerate(reg_loss):
                        objective_values[idx] = objective_values[idx] + reg

                total_objective = sum(objective_values)
                if total_objective.requires_grad:
                    candidate_grads = torch.autograd.grad(total_objective, candidates, create_graph=False)
                    grad_list = list(candidate_grads)
                    with torch.no_grad():
                        for idx, candidate_grad in enumerate(grad_list):
                            if self.cfg.optim.langevin_noise > 0:
                                step_size = optimizers[idx].param_groups[0]["lr"]
                                noise_map = torch.randn_like(candidate_grad)
                                candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                            if self.cfg.optim.grad_clip is not None:
                                grad_norm = candidate_grad.norm()
                                if grad_norm > self.cfg.optim.grad_clip:
                                    candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                            if self.cfg.optim.signed is not None:
                                if self.cfg.optim.signed == "soft":
                                    scaling_factor = 1 - iteration / self.cfg.optim.max_iterations
                                    candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                                elif self.cfg.optim.signed == "hard":
                                    candidate_grad.sign_()
                    torch.autograd.backward(candidates, grad_list)

                for optimizer, scheduler in zip(optimizers, schedulers):
                    optimizer.step()
                    scheduler.step()

                with torch.no_grad():
                    for idx, candidate in enumerate(candidates):
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        if objective_values[idx] < minimal_values_so_far[idx]:
                            minimal_values_so_far[idx] = objective_values[idx].detach()
                            best_candidates[idx] = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    for idx, (objective_value, task_loss) in enumerate(zip(objective_values, task_losses)):
                        log.info(
                            f"| Trial: {idx} | It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                    current_wallclock = timestamp

                if not all(torch.isfinite(obj) for obj in objective_values):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break
                for trial in range(num_trials):
                    stats[f"Trial_{trial}_Val"].append(objective_values[trial].item())
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        for netG in netGs:
            for p in netG.parameters():
                p.requires_grad = True
        noises = [noise.requires_grad_(False) for noise in noises]

        with torch.no_grad():
            best_candidates = [self._generate_candidate(netG, noise, labels).detach().clone() for netG, noise in zip(netGs, noises)]
        minimal_values_so_far = torch.as_tensor([float("inf")] * num_trials, **self.setup)

        prev_optim = {
            "optimizer": self.cfg.optim.optimizer,
            "step_size": self.cfg.optim.step_size,
            "max_iterations": self.cfg.optim.max_iterations,
            "callback": self.cfg.optim.callback,
        }
        self.cfg.optim.optimizer = self.cfg.optim_w.optimizer
        self.cfg.optim.step_size = self.cfg.optim_w.step_size
        self.cfg.optim.max_iterations = self.cfg.optim_w.max_iterations
        self.cfg.optim.callback = self.cfg.optim_w.callback

        optimizers, schedulers = [], []
        for netG in netGs:
            params = [p for p in netG.parameters() if p.requires_grad]
            optimizer, scheduler = self._init_optimizer(params)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                objective_values, task_losses, candidates = [], [], []
                candidate_for_models = []
                for netG, noise in zip(netGs, noises):
                    candidate = self._generate_candidate(netG, noise, labels)
                    if self.cfg.differentiable_augmentations:
                        candidate_augmented = self.augmentations(candidate)
                    else:
                        candidate_augmented = candidate
                        candidate_augmented.data = self.augmentations(candidate.data)
                    candidate_for_model = self._map_generator_output_to_model_space(candidate_augmented)
                    candidate_for_models.append(candidate_for_model)

                    total_objective = 0
                    total_task_loss = 0
                    for model, data in zip(rec_model, shared_data):
                        objective, task_loss = self.objective(model, data["gradients"], candidate_for_model, labels)
                        total_objective += objective
                        total_task_loss += task_loss
                    for regularizer in self.regularizers:
                        if regularizer.__class__.__name__ in ["AdversarialDiscriminatorRegularization"]:
                            total_objective += regularizer(candidate_augmented)
                        else:
                            total_objective += regularizer(candidate_for_model)

                    objective_values.append(total_objective)
                    task_losses.append(total_task_loss)
                    candidates.append(candidate)

                for regularizer in parallel_regularizers:
                    reg_loss = regularizer(candidate_for_models)
                    for idx, reg in enumerate(reg_loss):
                        objective_values[idx] = objective_values[idx] + reg

                total_objective = sum(objective_values)
                if total_objective.requires_grad:
                    candidate_grads = torch.autograd.grad(total_objective, candidates, create_graph=False)
                    grad_list = list(candidate_grads)
                    with torch.no_grad():
                        for idx, candidate_grad in enumerate(grad_list):
                            if self.cfg.optim.langevin_noise > 0:
                                step_size = optimizers[idx].param_groups[0]["lr"]
                                noise_map = torch.randn_like(candidate_grad)
                                candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                            if self.cfg.optim.grad_clip is not None:
                                grad_norm = candidate_grad.norm()
                                if grad_norm > self.cfg.optim.grad_clip:
                                    candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                            if self.cfg.optim.signed is not None:
                                if self.cfg.optim.signed == "soft":
                                    scaling_factor = 1 - iteration / self.cfg.optim.max_iterations
                                    candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                                elif self.cfg.optim.signed == "hard":
                                    candidate_grad.sign_()
                    torch.autograd.backward(candidates, grad_list)

                for optimizer, scheduler in zip(optimizers, schedulers):
                    optimizer.step()
                    scheduler.step()

                with torch.no_grad():
                    for idx, candidate in enumerate(candidates):
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        if objective_values[idx] < minimal_values_so_far[idx]:
                            minimal_values_so_far[idx] = objective_values[idx].detach()
                            best_candidates[idx] = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    for idx, (objective_value, task_loss) in enumerate(zip(objective_values, task_losses)):
                        log.info(
                            f"| Trial: {idx} | It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                    current_wallclock = timestamp

                if not all(torch.isfinite(obj) for obj in objective_values):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break
                for trial in range(num_trials):
                    stats[f"Trial_{trial}_Val"].append(objective_values[trial].item())
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        finally:
            self.cfg.optim.optimizer = prev_optim["optimizer"]
            self.cfg.optim.step_size = prev_optim["step_size"]
            self.cfg.optim.max_iterations = prev_optim["max_iterations"]
            self.cfg.optim.callback = prev_optim["callback"]

        self.netG = None
        return [candidate.detach() for candidate in best_candidates]

    def _compute_objective(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            # GENERATION
            if self.cfg.generator.type == "biggan":
                label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                candidate = netG(noise, label, truncation=self.truncation)
            elif self.cfg.generator.type == "stylegan_xl":
                label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type == "dcgan":
                candidate = netG(noise)
            candidate = self._resize_dcgan_output(candidate)
            # log.info(f"candidate.shape = {candidate.shape}")

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)
            candidate_for_model = self._map_generator_output_to_model_space(candidate_augmented)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                # data["gradients"] is a list of gradients for each layer
                objective, task_loss = self.objective(model, data["gradients"], candidate_for_model, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                    if regularizer.__class__.__name__ in ["AdversarialDiscriminatorRegularization"]:
                        total_objective += regularizer(candidate_augmented)
                    else:
                        total_objective += regularizer(candidate_for_model)

            if total_objective.requires_grad:
                candidate_grad = torch.autograd.grad(total_objective, [p for p in netG.parameters() if p.requires_grad], create_graph=False)
                with torch.no_grad():
                    if self.cfg.optim.langevin_noise > 0:
                        step_size = optimizer.param_groups[0]["lr"]
                        noise_map = torch.randn_like(candidate_grad)
                        candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                    if self.cfg.optim.grad_clip is not None:
                        grad_norm = candidate_grad.norm()
                        if grad_norm > self.cfg.optim.grad_clip:
                            candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                    if self.cfg.optim.signed is not None:
                        if self.cfg.optim.signed == "soft":
                            scaling_factor = (
                                1 - iteration / self.cfg.optim.max_iterations
                            )  # just a simple linear rule for now
                            candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                        elif self.cfg.optim.signed == "hard":
                            candidate_grad.sign_()
                        else:
                            pass
                
                for p, g in zip([p for p in netG.parameters() if p.requires_grad], candidate_grad):
                    p.grad = g

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure
    
    def _compute_objective_latent(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            if self.cfg.generator.type == "biggan":
                label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                candidate = netG(noise, label, truncation=self.truncation)
            elif self.cfg.generator.type == "stylegan_xl":
                label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type == "dcgan":
                candidate = netG(noise)
            candidate = self._resize_dcgan_output(candidate)
            # log.info(f"candidate.shape = {candidate.shape}")

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)
            candidate_for_model = self._map_generator_output_to_model_space(candidate_augmented)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):

                objective, task_loss = self.objective(model, data["gradients"], candidate_for_model, labels)
                total_objective += objective
                total_task_loss += task_loss
            
            # Add regularization only in the later stages of optimization, to allow more freedom in the initial stages
            for regularizer in self.regularizers:
                if regularizer.__class__.__name__ in ["AdversarialDiscriminatorRegularization"]:
                     total_objective += regularizer(candidate_augmented)
                else:
                    total_objective += regularizer(candidate_for_model)

            if total_objective.requires_grad:
                candidate_grad = torch.autograd.grad(total_objective, noise, create_graph=False)[0]
                with torch.no_grad():
                    if self.cfg.optim.langevin_noise > 0:
                        step_size = optimizer.param_groups[0]["lr"]
                        noise_map = torch.randn_like(candidate_grad)
                        candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                    if self.cfg.optim.grad_clip is not None:
                        grad_norm = candidate_grad.norm()
                        if grad_norm > self.cfg.optim.grad_clip:
                            candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                    if self.cfg.optim.signed is not None:
                        if self.cfg.optim.signed == "soft":
                            scaling_factor = (
                                1 - iteration / self.cfg.optim.max_iterations
                            )  # just a simple linear rule for now
                            candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                        elif self.cfg.optim.signed == "hard":
                            candidate_grad.sign_()
                        else:
                            pass
                noise.backward(candidate_grad)
                # candidate_grad = torch.autograd.grad(total_objective, netG.parameters(), create_graph=False)
                # for p, g in zip(netG.parameters(), candidate_grad):
                #     p.grad = g

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    
