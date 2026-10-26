"""Various objective functions that can be re-used for multiple attacks."""


import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import inspect

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from typing import List

from .make_functional import make_functional_with_buffers

from tqdm import tqdm

import logging

log = logging.getLogger(__name__)

class GradientLoss(torch.nn.Module):
    """Super-class to simplify gradient-based objectives."""

    def __init__(self):
        super().__init__()
        self.task_regularization = 0
        self.trainable_indices = None

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is None:
            self._grad_fn = self._grad_fn_single_step
        else:
            self._grad_fn = self._grad_fn_multi_step

        self.cfg_impl = cfg_impl


    def forward(self, model, gradient_data, candidate, labels):
        gradient, task_loss = self._grad_fn(model, candidate, labels)

        # Keep the target gradients on the same device/dtype as the reconstructed gradients.
        # Some payloads originate on CPU even when reconstruction runs on CUDA.
        gradient_data = [
            data.to(device=rec.device, dtype=rec.dtype) if rec is not None and data is not None else data
            for rec, data in zip(gradient, gradient_data)
        ]

        param_names = [name for name, p in model.named_parameters() if p.requires_grad]

        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):

            sig = inspect.signature(self.gradient_based_loss)

            if "param_names" in sig.parameters:
                objective = self.gradient_based_loss(
                    gradient,
                    gradient_data,
                    param_names=param_names
                )
            else:
                objective = self.gradient_based_loss(
                    gradient,
                    gradient_data
                )

        if self.task_regularization != 0:
            objective += self.task_regularization * task_loss

        return objective, task_loss.detach()

    def gradient_based_loss(self, gradient_rec, gradient_data, param_names=None):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def _grad_fn_single_step(self, model, candidate, labels):
        """Compute a single gradient."""
        model.zero_grad()
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            params = list(model.parameters())
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            with sdpa_kernel([SDPBackend.MATH]):
                task_loss = self.loss_fn(model(candidate), labels)
        gradient = torch.autograd.grad(task_loss, params, create_graph=True)
        return gradient, task_loss

    def _grad_fn_multi_step(self, model, candidate, labels):
        """
        Compute differentiable local SGD updates.

        Optimizations:
            - No dict/set lookups
            - No repeated trainable index discovery
            - Local variable caching
            - foreach updates when possible
            - minimal Python overhead
        """

        model.zero_grad()

        func_model, params, buffers = make_functional_with_buffers(model)

        # list is easier to mutate than tuple
        params = list(params)

        # Cache frequently used values
        hp = self.local_hyperparams
        lr = hp["lr"]
        steps = hp["steps"]
        data_per_step = hp["data_per_step"]
        labels_list = hp["labels"]

        # Trainable parameter indices
        if self.trainable_indices is None:
            self.trainable_indices = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
            self.num_trainable = len(self.trainable_indices)

        if self.num_trainable == 0:
            self.trainable_indices = list(range(len(params)))

        # Save initial parameters once
        initial_params = [params[i].clone() for i in self.trainable_indices]

        seen_data_idx = 0

        for step in tqdm(range(steps)):

            data = candidate[seen_data_idx:seen_data_idx + data_per_step]
            seen_data_idx = (seen_data_idx + data_per_step) % candidate.shape[0]
            step_labels = labels_list[step].to(device=data.device, dtype=torch.long)

            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                with sdpa_kernel([SDPBackend.MATH]):
                    task_loss = self.loss_fn(func_model(tuple(params), buffers, data), step_labels)

            trainable_params = [params[i] for i in self.trainable_indices]

            step_grads = torch.autograd.grad(task_loss, trainable_params, create_graph=True) #, retain_graph=True)

            # -------------------------------------------------
            # Fast foreach update
            # -------------------------------------------------
            # try:
            scaled_grads = torch._foreach_mul(list(step_grads), lr)
            updated_params = torch._foreach_sub(trainable_params, scaled_grads)

            for idx, new_param in zip(self.trainable_indices, updated_params):
                params[idx] = new_param

            # except RuntimeError:
            #     # fallback
            #     for idx, grad in zip(self.trainable_indices, step_grads,):
            #         params[idx] = params[idx] - lr * grad

            if self.cfg_impl.get("ica_feature_reg", False):
                with torch.no_grad():
                    _ = model(data)

        gradient = [params[idx] - initial_param for idx, initial_param in zip(self.trainable_indices, initial_params)]

        return gradient, task_loss

class Euclidean(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.step = 0
        self.max_iters = kwargs.get("max_iters", None)
        if self.max_iters is not None:
            log.info(f"Euclidean regularizer max iters: {self.max_iters}")

    def gradient_based_loss(self, gradient_rec, gradient_data):
        self.step += 1
        if self.max_iters is not None and self.step > self.max_iters // 2:
            return self._euclidean(gradient_rec, gradient_data) * self.scale / 2
        return self._euclidean(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"Euclidean loss with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _euclidean(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            data = data.to(device=rec.device, dtype=rec.dtype)
            objective += (rec - data).pow(2).sum()
        return 0.5 * objective


class EuclideanTag(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors plus TAG regularizer

    from Deng et al., "TAG: Gradient Attack on Transformer-based Language Models"
    How to scale each layer is unclear to me based on the paper, so I am recycling decay schemes from
    the InvertingGradients repo.
    """

    def __init__(self, scale=1.0, task_regularization=0.0, tag_scale=0.1, scale_scheme="linear", **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.tag_scale = tag_scale
        self.scale_scheme = scale_scheme
        self.weights = None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        if self.weights is None:
            setup = dict(dtype=gradient_rec[0].dtype, device=gradient_rec[0].device)
            if self.scale_scheme == "linear":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup) / len(gradient_rec)
            elif self.scale_scheme == "exp":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup)
                weights = weights.softmax(dim=0)
                weights = weights / weights[0]
            else:
                weights = gradient_rec[0].new_ones(len(gradient_rec))
        return self._weighted_euclidean_l1(gradient_rec, gradient_data, weights, self.tag_scale) * self.scale

    def __repr__(self):
        return (
            f"Tag loss with scale={self.scale}, weight scheme {self.scale_scheme}, L1 scale {self.tag_scale} "
            f"and task reg={self.task_regularization}"
        )

    @staticmethod
    @torch.jit.script
    def _weighted_euclidean_l1(
        gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], weights: torch.Tensor, tag_scale: float,
    ):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data, weight in zip(gradient_rec, gradient_data, weights):
            objective += (rec - data).pow(2).sum() + tag_scale * weight * (rec - data).abs().sum()
        return 0.5 * objective


class L1Loss(GradientLoss):
    """Gradient matching based on the L1 distance of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._l1loss(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"L1 loss with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _l1loss(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).abs().sum()
        # return 0.5 * objective
        return objective


class CosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        if kwargs.get("max_iterations", None) is not None:
            self.max_iterations = kwargs["max_iterations"]
            log.info(f"Cosine similarity regularizer max iters: {kwargs['max_iterations']}")
        else:
            self.max_iterations = None
        self.steps = 0

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            rec_flat = rec.flatten()
            data_flat = data.flatten().to(device=rec_flat.device, dtype=rec_flat.dtype)
            scalar_product += (rec_flat * data_flat).sum()
            rec_norm += rec_flat.pow(2).sum()
            data_norm += data_flat.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective

    def forward(self, model, gradient_data, candidate, labels):
        gradient, task_loss = self._grad_fn(model, candidate, labels)

        # Keep the target gradients on the same device/dtype as the reconstructed gradients.
        # Some payloads originate on CPU even when reconstruction runs on CUDA.
        gradient_data = [
            data.to(device=rec.device, dtype=rec.dtype) if rec is not None and data is not None else data
            for rec, data in zip(gradient, gradient_data)
        ]
        # param_names = [name for name, p in model.named_parameters() if p.requires_grad]
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            objective = self.gradient_based_loss(
                gradient,
                gradient_data
            )
        if self.task_regularization != 0:
            objective += self.task_regularization * task_loss

        self.steps += 1
        if self.max_iterations is not None and self.steps == self.max_iterations // 2:
            self.steps = self.scale / 2

        return objective, task_loss.detach()


class CosineSimilarityLayerWise(GradientLoss):
    """Gradient matching based on cosine similarity computed per layer."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        # log.info(f"Computing layer-wise cosine similarity. Number of layers: {len(gradient_rec)}")
        # for rec, data in zip(gradient_rec, gradient_data):
        #     rec = rec.flatten()
        #     data = data.flatten()
        #     scalar_product = (rec * data).sum()
        #     rec_norm = rec.pow(2).sum() + 1e-4
        #     data_norm = data.pow(2).sum() + 1e-4
        #     objective = - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        #     log.info(f"Layer objective: {objective:.4f}, rec norm: {rec_norm:.4f}, data norm: {data_norm:.4f}")
        return self._cosine_sim_layerwise(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"Layer-wise Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _cosine_sim_layerwise(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            rec = rec.flatten()
            data = data.flatten()
            scalar_product = (rec * data).sum()
            rec_norm = rec.pow(2).sum() + 1e-4
            data_norm = data.pow(2).sum() + 1e-4
            objective += 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        objective /= len(gradient_rec)
        # objective = 1 - objective
        return objective

class EuclideanGradCosSimPosEmbed(GradientLoss):
    """Gradient matching based on cosine similarity of two sets of PosEmbeddings."""

    def __init__(self, scale=1.0, task_regularization=0.0, posembed_scale=0.1, weight_start=0.6, weight_end=1.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.posembed_scale = posembed_scale
        self.pop_index = kwargs.get("pop_index", 1)
        # vpt: 2, lora: 0, full: 1
        self.weight_start = weight_start
        self.weight_end = weight_end

    def _check_shapes(self, grads1, grads2, name=""):
        assert len(grads1) == len(grads2), f"{name}: Length mismatch {len(grads1)} vs {len(grads2)}"

        for i, (g1, g2) in enumerate(zip(grads1, grads2)):
            if g1 is None or g2 is None:
                continue
            #print(f"{name}: Layer {i} shapes: {g1.shape} vs {g2.shape}")
            if g1.shape != g2.shape:
                raise ValueError(f"{name}: Shape mismatch at index {i}: {g1.shape} vs {g2.shape}")
    def gradient_based_loss(self, gradient_rec, gradient_data, param_names=None):
        gradient_rec = list(gradient_rec)
        gradient_data = list(gradient_data)
        # --------------------------------------------------
        # 🔥 CASE 1: New dynamic splitting (preferred)
        # --------------------------------------------------

        if param_names is not None:
            pos_rec, pos_data = [], []
            main_rec, main_data = [], []
            for g_rec, g_data, name in zip(gradient_rec, gradient_data, param_names):
                if g_rec is None or g_data is None:
                    continue
                if "pos_embed" in name or "gamma" in name:
                    # THIS WORKS -- POSEMBED IS DETECTED CORRECTLY
                    # log.info(f"Identified pos_embed/gamma layer: {name} with shape {g_rec.shape}")
                    pos_rec.append(g_rec)
                    pos_data.append(g_data)
                else:
                    main_rec.append(g_rec)
                    main_data.append(g_data)
            #print(f"Identified {len(pos_rec)} pos_embed/gamma layers and {len(main_rec)} main layers based on parameter names.")
            # ---- Debug checks ----
            #self._check_shapes(main_rec, main_data, "main grads")
            for i, (pr, pd) in enumerate(zip(pos_rec, pos_data)):
                if pr.shape != pd.shape:
                    raise ValueError(f"pos_embed mismatch at {i}: {pr.shape} vs {pd.shape}")

            # ---- Loss computation ----
            main_loss = 0.0
            if len(main_rec) > 0:
                main_loss = self._euclidean(main_rec, main_data) * self.scale

            pos_loss = 0.0
            if len(pos_rec) > 0:
                pos_loss = self._cosine_sim(pos_rec, pos_data) * self.posembed_scale

            return main_loss + pos_loss

        # --------------------------------------------------
        # 🔁 CASE 2: Fallback (old behavior using pop index)
        # --------------------------------------------------
        else:
            # log.info(f"grad-rec: {len(gradient_rec)}, grad-data: {len(gradient_data)}")
            posembed_rec = gradient_rec.pop(self.pop_index)
            posembed_data = gradient_data.pop(self.pop_index)
            log.info(f"Shape of posembed_rec: {posembed_rec.shape}, posembed_data: {posembed_data.shape}")
            # log.info(f"grad-rec: {len(gradient_rec)}, grad-data: {len(gradient_data)}")
            euc_grad = self._euclidean(gradient_rec, gradient_data)
            pos_grad = self._cosine_sim([posembed_rec], [posembed_data])
            return euc_grad * self.scale + pos_grad * self.posembed_scale

    def __repr__(self):
        return f"Euclidean with scale={self.scale} and Cosine Similarity for PosEmbed with scale={self.posembed_scale} and task regularization {self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _euclidean(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
        return objective

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            rec_flat = rec.flatten()
            data_flat = data.flatten().to(device=rec_flat.device, dtype=rec_flat.dtype)
            scalar_product += (rec_flat * data_flat).sum()
            rec_norm += rec_flat.pow(2).sum()
            data_norm += data_flat.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective

class CosSimGradCosSimPosEmbed(GradientLoss):
    """Cosine similarity loss with separate handling for positional embeddings."""

    def __init__(self, scale=1.0, task_regularization=0.0, posembed_scale=0.1, weight_start=0.6, weight_end=1.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.posembed_scale = posembed_scale
        self.pop_index = kwargs.get("pop_index", 1)  # fallback only
        self.weight_start = weight_start
        self.weight_end = weight_end

    def _check_shapes(self, grads1, grads2, name=""):
        assert len(grads1) == len(grads2), f"{name}: Length mismatch {len(grads1)} vs {len(grads2)}"

        for i, (g1, g2) in enumerate(zip(grads1, grads2)):
            if g1 is None or g2 is None:
                continue
            #print(f"{name}: Layer {i} shapes: {g1.shape} vs {g2.shape}")
            if g1.shape != g2.shape:
                raise ValueError(f"{name}: Shape mismatch at index {i}: {g1.shape} vs {g2.shape}")

    def gradient_based_loss(self, gradient_rec, gradient_data, param_names=None):
        gradient_rec = list(gradient_rec)
        gradient_data = list(gradient_data)

        # --------------------------------------------------
        # 🔥 CASE 1: New dynamic splitting (preferred)
        # --------------------------------------------------

        if param_names is not None:

            pos_rec, pos_data = [], []
            main_rec, main_data = [], []

            for g_rec, g_data, name in zip(gradient_rec, gradient_data, param_names):

                if g_rec is None or g_data is None:
                    continue

                if "pos_embed" in name or "gamma" in name:
                    pos_rec.append(g_rec)
                    pos_data.append(g_data)
                else:
                    main_rec.append(g_rec)
                    main_data.append(g_data)
            #print(f"Identified {len(pos_rec)} pos_embed/gamma layers and {len(main_rec)} main layers based on parameter names.")

            # ---- Debug checks ----
            #self._check_shapes(main_rec, main_data, "main grads")

            for i, (pr, pd) in enumerate(zip(pos_rec, pos_data)):
                if pr.shape != pd.shape:
                    raise ValueError(f"pos_embed mismatch at {i}: {pr.shape} vs {pd.shape}")

            # ---- Loss computation ----
            main_loss = 0.0
            if len(main_rec) > 0:
                main_loss = self._cosine_sim(main_rec, main_data) * self.scale

            pos_loss = 0.0
            if len(pos_rec) > 0:
                pos_loss = self._cosine_sim(pos_rec, pos_data) * self.posembed_scale

            return main_loss + pos_loss

        # --------------------------------------------------
        # 🔁 CASE 2: Fallback (old behavior using pop index)
        # --------------------------------------------------
        else:
            posembed_rec = gradient_rec.pop(self.pop_index)
            posembed_data = gradient_data.pop(self.pop_index)

            self._check_shapes(gradient_rec, gradient_data, "main grads")

            if posembed_rec.shape != posembed_data.shape:
                raise ValueError(f"posembed mismatch: {posembed_rec.shape} vs {posembed_data.shape}")

            return (
                self._cosine_sim(gradient_rec, gradient_data) * self.scale
                + self._cosine_sim([posembed_rec], [posembed_data]) * self.posembed_scale
            )

    def __repr__(self):
        return (
            f"Cosine Similarity with scale={self.scale} and "
            f"Cosine Similarity for PosEmbed with scale={self.posembed_scale} "
            f"and task regularization {self.task_regularization}"
        )
    

    @staticmethod
    @torch.jit.script
    def _cosine_sim_layerwise(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for gr, gd in zip(gradient_rec, gradient_data):
            gr = gr.flatten()
            gd = gd.flatten().to(device=gr.device, dtype=gr.dtype)
            scalar_product = (gr * gd).sum()
            rec_norm = gr.pow(2).sum()
            data_norm = gd.pow(2).sum()

            objective += 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        objective /= len(gradient_rec)
        return objective
    
    @staticmethod
    def _align_gradients_by_shape(gradient_rec, gradient_data):
        """Align tensors by exact shape to avoid cross-layer mismatches."""
        remaining = list(gradient_data)
        aligned_rec, aligned_data = [], []
        for rec in gradient_rec:
            rec_shape = tuple(rec.shape)
            match_idx = None
            for idx, data in enumerate(remaining):
                if tuple(data.shape) == rec_shape:
                    match_idx = idx
                    break
            if match_idx is not None:
                aligned_rec.append(rec)
                aligned_data.append(remaining.pop(match_idx))
        return aligned_rec, aligned_data

    @staticmethod
    @torch.jit.script
    def _cosine_sim_weighted_layerwise(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor],weight_start: float, weight_end: float):
        num_layers = len(gradient_rec)
        device = gradient_rec[0].device
        dtype = gradient_rec[0].dtype

        # Linear weights from 0.6 → 1.0
        weights = torch.linspace(weight_start, weight_end, steps=num_layers, device=device, dtype=dtype)

        objective = gradient_rec[0].new_zeros(1,)
        eps = 1e-12

        for gr, gd, weight in zip(gradient_rec, gradient_data, weights):
            gr = gr.flatten()
            gd = gd.flatten()

            scalar_product = (gr * gd).sum()
            rec_norm = gr.pow(2).sum()
            data_norm = gd.pow(2).sum()

            rec_norm = rec_norm.clamp_min(eps)
            data_norm = data_norm.clamp_min(eps)
            cosine = scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
            cosine = torch.clamp(cosine, -1.0, 1.0)

            objective += weight * (1 - cosine)

        # Proper weighted normalization
        objective /= weights.sum()

        return objective
    
    @staticmethod
    @torch.jit.script
    def _cosine_sim_weighted_global(
        gradient_rec: List[torch.Tensor],
        gradient_data: List[torch.Tensor],
        weight_start: float,
        weight_end: float
    ):
        num_layers = len(gradient_rec)
        device = gradient_rec[0].device
        dtype = gradient_rec[0].dtype

        weights = torch.linspace(weight_start, weight_end, steps=num_layers, device=device, dtype=dtype)

        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        eps = 1e-12

        for rec, data, w in zip(gradient_rec, gradient_data, weights):
            rec_w = w * rec
            data_w = w * data

            scalar_product += (rec_w * data_w).sum()
            rec_norm += rec_w.pow(2).sum()
            data_norm += data_w.pow(2).sum()

        rec_norm = rec_norm.clamp_min(eps)
        data_norm = data_norm.clamp_min(eps)

        cosine = scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        cosine = torch.clamp(cosine, -1.0, 1.0)

        return 1 - cosine

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            rec_flat = rec.flatten()
            data_flat = data.flatten().to(device=rec_flat.device, dtype=rec_flat.dtype)
            scalar_product += (rec_flat * data_flat).sum()
            rec_norm += rec_flat.pow(2).sum()
            data_norm += data_flat.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective

class AngularSimilarity(CosineSimilarity):
    """Gradient matching based on angular similarity of two gradient vectors.

    This is basically a more linear cosine similarity."""

    def __init__(self, scale=1.0, task_regularization=0.0, fudge_factor=1e-7, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.fudge_factor = 1e-7

    def gradient_based_loss(self, gradient_rec, gradient_data):
        cosine = 1 - self._cosine_sim(gradient_rec, gradient_data)
        angle = torch.acos(cosine.clamp(min=-1 + self.fudge_factor, max=1 - self.fudge_factor))

        return angle / torch.pi * self.scale

    def __repr__(self):
        return f"Angular Similarity with scale={self.scale} and task reg={self.task_regularization}"


class MaskedCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    All positions that are zero in the data gradient are masked.
    """

    def __init__(self, scale=1.0, mask_value=1e-6, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.mask_value = 1e-6
        self.task_regularization = task_regularization

    def __repr__(self):
        return f"Masked Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}. Mask val={self.mask_value}"

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            mask = data.abs() > self.mask_value
            scalar_product += (rec * data * mask).sum()
            rec_norm += (rec * mask).pow(2).sum()
            data_norm += (data * mask).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class FastCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    No gradient flows through the normalization."""

    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective

    def __repr__(self):
        return f"Fast Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"


class PearlmutterEuclidean(torch.nn.Module):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def __init__(
        self,
        scale=1.0,
        eps=1e-3,
        level_gradients=False,
        fudge_factor=1e-6,
        task_regularization=0.0,
        implementation="forward",
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

        self.eps = eps
        self.level_gradients = level_gradients
        self.fudge_factor = fudge_factor
        self.implementation = implementation

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is not None:
            raise ValueError("This loss is only implemented for local gradients so far.")

        self.cfg_impl = cfg_impl
        if self.implementation == "forward":
            self._forward_impl = self._forward_differences
        elif self.implementation == "backward":
            self._forward_impl = self._backward_differences
        elif self.implementation == "central":
            self._forward_impl = self._central_differences
        elif self.implementation == "upwind":
            self._forward_impl = self._upwind_differences
        else:
            raise ValueError(f"Invalid finite difference implementation {self.implementation} given.")

    def __repr__(self):
        return (
            f"Pearlmutter-type Finite Differences Loss with scale={self.scale} and task reg={self.task_regularization}."
            f"Finite Difference Eps: {self.eps}. Level gradients: {self.level_gradients}. "
            f"{f'Fudge-factor: {self.fudge_factor}' if self.level_gradients else ''}"
        )

    def forward(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        model.zero_grad()
        # Keep original parameters
        original_parameters = [p.detach().clone() for p in model.parameters()]
        # Compute derivative and accumulate into candidate.grad
        objective_value, task_loss = self._forward_impl(model, gradient_data, candidate, labels)

        # Unpatch model:# Want to do this faster, somehow
        for param, original_param in zip(model.parameters(), original_parameters):
            param.data.copy_(original_param)
        return objective_value, task_loss

    def _forward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model_params_with_grad, candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLv_dx - dLdx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _backward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model_params_with_grad, candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLdx - dLv_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _central_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model_params_with_grad, candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLvp_dx - dLvm_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _upwind_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model_params_with_grad, candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute both finite differences
        Dp = (dLvp_dx - dLdx) / eps_n
        Dm = (dLdx - dLvm_dx) / eps_n
        # Upwind based on dLdx
        # Compute finite difference approximation
        candidate.grad += (torch.max(dLdx, 0)[0] * Dm + torch.min(dLdx, 0)[0] * Dp) * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        residuals = torch._foreach_sub(gradients, gradient_data)  # Save one copy of the gradient list here ?
        # Gradients have already been populated. Make sure not to kill them later on.
        with torch.no_grad():
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                objective_value = 0.5 * self.scale * torch.stack([r.detach().pow(2).sum() for r in residuals]).sum()
        return objective_value, residuals


class PearlmutterCosine(PearlmutterEuclidean):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        with torch.no_grad():
            scalar_product, grad_norm, data_norm = self._cosine_sim_components(gradients, gradient_data)

        first_order_cosine = torch._foreach_div(gradient_data, -grad_norm * data_norm)
        torch._foreach_sub_(first_order_cosine, gradients, alpha=-scalar_product / (grad_norm.pow(3) * data_norm))

        objective_value = self.scale * (1 - scalar_product / (grad_norm * data_norm))
        return objective_value, first_order_cosine

    @staticmethod
    @torch.jit.script
    def _cosine_sim_components(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        rec_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        data_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        return scalar_product, rec_norm.sqrt(), data_norm.sqrt()
    

class BoostingGLAEuclidean(torch.nn.Module):
    """
    FEDLEAK-style perturb-and-blend using candidate-space perturbations.
    """

    def __init__(
        self,
        scale=1.0,
        R=0.5,
        lambda_blend=0.7,
        eps=1e-3,
        task_regularization=0.0,
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.R = R
        self.lambda_blend = lambda_blend
        self.eps = eps
        self.task_regularization = task_regularization

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.cfg_impl = cfg_impl
        if local_hyperparams is not None:
            raise ValueError("This loss only supports local gradients.")

    def __repr__(self):
        return (
            f"PearlmutterBoostingGLA(scale={self.scale}, R={self.R}, "
            f"lambda_blend={self.lambda_blend}, eps={self.eps})"
        )

    def forward(self, model, gradient_data, candidate, labels):
        model.zero_grad()

        # ---- FIRST PASS: x′ ----
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)

        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients, dLdx = torch.autograd.grad(
            task_loss,
            (*model_params_with_grad, candidate),
            create_graph=True
        )

        # partial gradient matching + L1+cosine
        D, grad_x = self._gradient_matching_and_grad(
            candidate, gradients, gradient_data
        )

        phi = self.eps * grad_x / (grad_x.norm() + 1e-12)

        candidate_pert = (candidate + phi).detach().requires_grad_(True)

        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss_p = self.loss_fn(model(candidate_pert), labels)

        model_params_with_grad = [p for p in model.parameters() if p.requires_grad]
        *gradients_p, dLdx_p = torch.autograd.grad(
            task_loss_p,
            (*model_params_with_grad, candidate_pert),
            create_graph=True
        )

        _, grad_x_p = self._gradient_matching_and_grad(
            candidate_pert, gradients_p, gradient_data
        )

        # Blend
        candidate.grad = (
            (1.0 - self.lambda_blend) * grad_x
            + self.lambda_blend * grad_x_p
        ) * self.scale

        # optional task loss regularization
        candidate.grad += self.task_regularization * dLdx

        return D.detach(), task_loss.detach()

    def _gradient_matching_and_grad(self, candidate, gradients, gradient_data):
        """
        Computes:
        - Partial gradient matching (top-R%)
        - L1 + cosine distance
        - ∇x′ D(x′)
        """

        mags = torch.tensor(
            [g.abs().sum() for g in gradients],
            device=gradients[0].device
        )

        k_top = max(1, int(self.R * mags.numel()))
        indices = torch.topk(mags, k_top, largest=True).indices

        grad_sel = [gradients[i] for i in indices]
        data_sel = [gradient_data[i] for i in indices]

        # L1 + cosine dist.
        l1 = grad_sel[0].new_zeros(1,)
        dot = grad_sel[0].new_zeros(1,)
        norm_g = grad_sel[0].new_zeros(1,)
        norm_d = grad_sel[0].new_zeros(1,)

        for g, d in zip(grad_sel, data_sel):
            l1 += (g - d).abs().sum()
            dot += (g * d).sum()
            norm_g += g.pow(2).sum()
            norm_d += d.pow(2).sum()

        cosine = dot / (torch.sqrt(norm_g) * torch.sqrt(norm_d) + 1e-12)
        D = l1 + (1.0 - cosine)

        grad_x = torch.autograd.grad(
            D,
            candidate,
            retain_graph=False
        )[0]

        return D, grad_x



objective_lookup = {
    "euclidean": Euclidean,
    "cosine-similarity": CosineSimilarity,
    "layerwise-cosine-similarity": CosineSimilarityLayerWise,
    "euclidean-grad-cossim-posembed": EuclideanGradCosSimPosEmbed,
    "cosine-grad-cossim-posembed": CosSimGradCosSimPosEmbed,
    "masked-cosine-similarity": MaskedCosineSimilarity,
    "fast-cosine-similarity": FastCosineSimilarity,
    "angular": AngularSimilarity,
    "l1": L1Loss,
    "pearlmutter-loss": PearlmutterEuclidean,
    "pearlmutter-cosine": PearlmutterCosine,
    "tag-euclidean": EuclideanTag,
    "boosting-euclidean": BoostingGLAEuclidean
}
