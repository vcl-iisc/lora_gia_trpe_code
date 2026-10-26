"""Helper code to instantiate various models."""

import math
import logging
from pathlib import Path
import torch
import torchvision
import torch.nn as nn

from collections import OrderedDict

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

from .language_models import RNNModel, TransformerModel, LinearModel
from .losses import CausalLoss, MLMLoss, MostlyCausalLoss

log = logging.getLogger(__name__)





def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


def _model_name(cfg_model):
    if isinstance(cfg_model, str):
        return cfg_model
    name = _cfg_get(cfg_model, "name", None)
    if name is None:
        raise ValueError("Model config must provide a model name.")
    return name


def _resolve_vit_image_size(cfg_model, cfg_data):
    image_size = _cfg_get(cfg_data, "image_size", None)
    if image_size is None:
        image_size = _cfg_get(cfg_model, "image_size", None)
    if image_size is None and hasattr(cfg_data, "shape"):
        shape = _cfg_get(cfg_data, "shape", None)
        if shape is not None and len(shape) >= 3:
            image_size = (int(shape[1]), int(shape[2]))
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 1:
            value = int(image_size[0])
            return (value, value)
        if len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
    raise ValueError("ViT image_size must be an int or a 2-tuple/list.")


def _resolve_vit_patch_size(cfg_model, cfg_data, default_patch_size=16):
    patch_size = _cfg_get(cfg_model, "patch_size", None)
    if patch_size is None:
        patch_size = _cfg_get(cfg_data, "patch_size", None)
    if patch_size is None:
        return int(default_patch_size)
    if isinstance(patch_size, int):
        return int(patch_size)
    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 1:
            return int(patch_size[0])
        if len(patch_size) == 2 and int(patch_size[0]) == int(patch_size[1]):
            return int(patch_size[0])
        return tuple(int(v) for v in patch_size)
    return int(patch_size)


def _load_checkpoint_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise ValueError(f"Pretrained checkpoint path must point to a file: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            candidate = state_dict.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}: {type(checkpoint)!r}")

    for prefix in ("module.", "model."):
        if len(state_dict) > 0 and all(isinstance(key, str) and key.startswith(prefix) for key in state_dict.keys()):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}

    return state_dict


def _load_pretrained_weights(model, pretrained_path):
    if pretrained_path is None:
        log.info("Pretrained Path is None, Returning Model with Random Initialization.")
        return model

    state_dict = _load_checkpoint_state_dict(pretrained_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        log.warning(
            "Loaded checkpoint %s with missing keys %s and unexpected keys %s",
            pretrained_path,
            missing_keys,
            unexpected_keys,
        )
    log.info("Loaded pretrained weights from %s", pretrained_path)
    return model


def _is_lora_one_init(init_name):
    return str(init_name).lower() in {"lora_one", "lora-one", "loraone"}


def _normalize_lora_init(init_name):
    init_key = str(init_name).lower()
    if init_key in {"lora_one", "lora-one", "loraone"}:
        return "lora_one"
    if init_key == "lora_ga":
        return "lora_ga"
    return init_name


def _select_lora_one_probe_user_idx(cfg_case):
    cfg_user = _cfg_get(cfg_case, "user", None) if cfg_case is not None else None
    cfg_data = _cfg_get(cfg_case, "data", None) if cfg_case is not None else None
    if cfg_user is None or cfg_data is None:
        return None

    target_user_idx = int(_cfg_get(cfg_user, "user_idx", 0))
    default_clients = int(_cfg_get(cfg_data, "default_clients", 1))
    if default_clients < 2:
        log.warning("LoRA-One probe batch needs at least two user partitions for held-out sampling.")
        return None

    partition = str(_cfg_get(cfg_data, "partition", ""))
    if partition == "none":
        log.warning("LoRA-One probe batch cannot be held out when the dataset partition is `none`.")
        return None
    if partition == "random-full":
        log.warning("LoRA-One probe batch cannot be guaranteed out-of-sample under `random-full` partition.")
        return None

    peft_cfg = _cfg_get(cfg_case, "peft", None)
    lora_cfg = _cfg_get(peft_cfg, "lora", peft_cfg)
    explicit_probe_user = _cfg_get(lora_cfg, "probe_user_idx", None)
    if explicit_probe_user is not None:
        probe_user_idx = int(explicit_probe_user)
    else:
        probe_user_offset = int(_cfg_get(lora_cfg, "probe_user_offset", 1))
        probe_user_idx = (target_user_idx + probe_user_offset) % default_clients
        if probe_user_idx == target_user_idx:
            probe_user_idx = (target_user_idx + 1) % default_clients

    if probe_user_idx == target_user_idx:
        log.warning("LoRA-One probe user must differ from the attacked user.")
        return None
    return probe_user_idx


def _load_lora_one_probe_batch(cfg_case, device):
    if cfg_case is None:
        return None
    try:
        from ..data import construct_dataloader  # Lazy import to avoid startup circularity.
    except Exception as error:
        log.warning("LoRA-One probe batch construction failed (import): %s", error)
        return None

    peft_cfg = _cfg_get(cfg_case, "peft", None)
    lora_cfg = _cfg_get(peft_cfg, "lora", peft_cfg)
    requested_points = int(_cfg_get(lora_cfg, "probe_batch_size", 8))
    probe_user_idx = _select_lora_one_probe_user_idx(cfg_case)
    if requested_points <= 0 or probe_user_idx is None:
        return None
    try:
        # Use a held-out user partition so adapter initialization does not depend on the attacked
        # user's own sample or batch size.
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=probe_user_idx, return_full_dataset=False)
    except Exception as error:
        log.warning("LoRA-One probe batch construction failed (dataloader): %s", error)
        return None

    data_blocks = []
    num_samples = 0
    for block in dataloader:
        data_blocks.append(block)
        num_samples += int(block["labels"].shape[0])
        if num_samples >= requested_points:
            break

    if len(data_blocks) == 0:
        log.warning("LoRA-One probe batch is empty, falling back to default LoRA init.")
        return None

    batch = {}
    for key in data_blocks[0]:
        batch[key] = torch.cat([item[key] for item in data_blocks], dim=0)[:requested_points].to(device=device)
    return batch


def _default_lora_one_gamma(cfg_case):
    return 128.0


def _compute_lora_one_weight_grads(backbone, peft_cfg, cfg_case, target_set):
    lora_cfg = _cfg_get(peft_cfg, "lora", peft_cfg)
    probe_batch = _load_lora_one_probe_batch(cfg_case, next(backbone.parameters()).device)
    if probe_batch is None or "labels" not in probe_batch:
        return {}

    probe_input = probe_batch.get("inputs", probe_batch.get("input_ids", None))
    if probe_input is None:
        log.warning("LoRA-One requires `inputs` or `input_ids` in probe batch; falling back to default init.")
        return {}

    target_params = {}
    for block in backbone.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear) and any(t in target_set for t in ("q", "k", "v")):
                target_params[id(attn.qkv.weight)] = attn.qkv.weight
            else:
                if "q" in target_set and hasattr(attn, "q_proj") and isinstance(attn.q_proj, nn.Linear):
                    target_params[id(attn.q_proj.weight)] = attn.q_proj.weight
                if "k" in target_set and hasattr(attn, "k_proj") and isinstance(attn.k_proj, nn.Linear):
                    target_params[id(attn.k_proj.weight)] = attn.k_proj.weight
                if "v" in target_set and hasattr(attn, "v_proj") and isinstance(attn.v_proj, nn.Linear):
                    target_params[id(attn.v_proj.weight)] = attn.v_proj.weight
            if "proj" in target_set and hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                target_params[id(attn.proj.weight)] = attn.proj.weight

        if "mlp" in target_set:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                    target_params[id(mlp.fc1.weight)] = mlp.fc1.weight
                if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                    target_params[id(mlp.fc2.weight)] = mlp.fc2.weight

    if len(target_params) == 0:
        return {}

    previous_training = backbone.training
    backbone.train()
    try:
        logits = backbone(probe_input)
        labels = probe_batch["labels"].to(device=logits.device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        ordered_params = list(target_params.values())
        grads = torch.autograd.grad(loss, ordered_params, allow_unused=True)
    except Exception as error:
        log.warning("LoRA-One probe gradient computation failed: %s. Falling back to default LoRA init.", error)
        return {}
    finally:
        backbone.train(previous_training)

    grad_map = {}
    for param, grad in zip(target_params.values(), grads):
        if grad is not None and grad.ndim == 2:
            grad_map[id(param)] = grad.detach()
    if len(grad_map) == 0:
        log.warning("LoRA-One found no usable 2D gradients; falling back to default LoRA init.")
    return grad_map

def lora_one_init_from_grad(grad_weight: torch.Tensor, rank: int, gamma: float = 1.0):
    """
    LoRA-One spectral initialization from one-step full gradient.

    Args:
        grad_weight: gradient of weight matrix [out_features, in_features]
        rank: LoRA rank
        gamma: scaling hyperparameter

    Returns:
        A: [rank, in_features]
        B: [out_features, rank]
    """
    # Current A and B are taken from the right components, but CONFUSING!!!
    # I think grad_weight should also be transposed before doing SVD
    # Transposed now. But need to transpose again later
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(-grad_weight.T, full_matrices=False)

        r = min(rank, S.shape[0])

        U_r = U[:, :r]  # [in, r]
        S_r = S[:r]  # [r]
        V_r = Vh[:r, :]  # [r, out]

        # Normalize singular values
        S_r = S_r / S_r[0]

        if gamma <= 0:
            raise ValueError("LoRA-One gamma must be positive.")
        gamma_sqrt = math.sqrt(float(gamma))
        S_half = torch.sqrt(S_r)
        S_half_diag = torch.diag(S_half)

        # LoRA injects `B @ A` additively into the frozen weight path.
        # Under RS-LoRA scaling (alpha / sqrt(rank)), this gives the best rank-r one-step
        # descent direction with effective step size alpha / (gamma * sqrt(rank)).
        A_r = (U_r @ S_half_diag) * gamma_sqrt  # [in, r]
        B_r = (S_half_diag @ V_r) * gamma_sqrt  # [r, out]

        # A = torch.zeros(rank, grad_weight.shape[1], device=grad_weight.device, dtype=grad_weight.dtype)
        # B = torch.zeros(grad_weight.shape[0], rank, device=grad_weight.device, dtype=grad_weight.dtype)
        # A[:r, :] = A_r
        # B[:, :r] = B_r
        return A_r.to( device=grad_weight.device, dtype=grad_weight.dtype), B_r.to( device=grad_weight.device, dtype=grad_weight.dtype)


class LoRALinear(nn.Module):
    """LoRA wrapper for a linear layer."""

    def __init__(
        self,
        base_layer,
        rank=4,
        alpha=1.0,
        dropout=0.0,
        init="standard",
        grad_weight=None,
        lora_one_gamma=2.0,
        use_rslora=False,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects a torch.nn.Linear layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.use_rslora = bool(use_rslora)
        self.scaling = float(alpha) / math.sqrt(float(rank)) if self.use_rslora else float(alpha) / math.sqrt(float(rank))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        init = _normalize_lora_init(init)

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))

        def adjust_base_weight(sign):
            with torch.no_grad():
                delta = self.lora_B @ self.lora_A
                self.base_layer.weight.data.add_(sign * delta)
        
        if init == "normal":
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
            adjust_base_weight(-self.scaling)
        elif init == "uniform":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            adjust_base_weight(-self.scaling)
        elif init == "standard":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)
        elif init == "lora_ga":
            pass
        elif init == "lora_one":
            if grad_weight is None:
                # fallback placeholder initialization
                nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
                nn.init.zeros_(self.lora_B)
            else:
                A, B = lora_one_init_from_grad(
                    grad_weight.detach().to(device=self.lora_A.device, dtype=self.lora_A.dtype),
                    rank=self.rank,
                    gamma=float(lora_one_gamma), #float(lora_one_gamma),
                )
                self.lora_A.data.copy_(A.T)
                self.lora_B.data.copy_(B.T)
                adjust_base_weight(self.scaling)
        else:
            raise ValueError(
                f"Unknown LoRA initialization: {init}. Choose from 'normal', 'uniform', 'standard', 'lora_one'."
            )

    def forward(self, x):
        out = self.base_layer(x)
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T # mathematically correct
        return out + delta * self.scaling

    def shared_parameters(self):
        return [self.lora_A, self.lora_B]

from timm.layers import PatchEmbed

class LinearPatchEmbed(PatchEmbed):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=384, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)
    
    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        x = x.reshape(x.size(0), self.num_patches, -1)
        x = self.proj(x) # B, num_patches, embed_dim
        return x

class LoRAQKVLinear(nn.Module):
    """LoRA wrapper for fused qkv linear layers in timm ViT blocks."""

    def __init__(
        self,
        base_layer,
        rank=4,
        alpha=1.0,
        dropout=0.0,
        apply_q=True,
        apply_k=True,
        apply_v=True,
        init="standard",
        grad_weight_q=None,
        grad_weight_k=None,
        grad_weight_v=None,
        lora_one_gamma=2.0,
        use_rslora=False,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRAQKVLinear expects a torch.nn.Linear layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if base_layer.out_features % 3 != 0:
            raise ValueError("Fused qkv layer must have out_features divisible by 3.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.use_rslora = bool(use_rslora)
        self.scaling = float(alpha) / (math.sqrt(float(rank)) if self.use_rslora else float(rank))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.chunk = base_layer.out_features // 3
        self.has_any_adapter = bool(apply_q or apply_k or apply_v)
        init = _normalize_lora_init(init)

        def make_pair(enabled, init, grad_weight, start_row):
            if not enabled:
                return None, None
            A = nn.Parameter(torch.zeros(self.rank, base_layer.in_features))
            B = nn.Parameter(torch.zeros(self.chunk, self.rank))

            def adjust_base_weight(sign):
                with torch.no_grad():
                    delta = B @ A
                    self.base_layer.weight.data[start_row : start_row + self.chunk].add_(sign * delta)

            if init == "normal":
                nn.init.normal_(A, mean=0.0, std=0.02)
                nn.init.normal_(B, mean=0.0, std=0.02)
                adjust_base_weight(-self.scaling)
            elif init == "uniform":
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                nn.init.kaiming_uniform_(B, a=math.sqrt(5))
                adjust_base_weight(-self.scaling)
            elif init == "standard":
                # nn.init.normal_(A, mean=0.0, std=0.02)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                nn.init.zeros_(B)
            elif init == "lora_ga":
                pass
            elif init == "lora_one":
                if grad_weight is None:
                    # fallback placeholder initialization
                    nn.init.normal_(A, mean=0.0, std=0.02)
                    nn.init.zeros_(B)
                else:
                    A_init, B_init = lora_one_init_from_grad(
                        grad_weight.detach().to(device=A.device, dtype=A.dtype),
                        rank=self.rank,
                        gamma=float(lora_one_gamma),
                    )
                    A.data.copy_(A_init.T)
                    B.data.copy_(B_init.T)
                    adjust_base_weight(self.scaling)  # LoRA-One is designed to be added with the specified scaling factor.
            else:
                raise ValueError(
                    f"Unknown LoRA initialization: {init}. Choose from 'normal', 'uniform', 'standard', 'lora_one'."
                )
            return A, B

        self.lora_A_q, self.lora_B_q = make_pair(apply_q, init, grad_weight_q, 0)
        self.lora_A_k, self.lora_B_k = make_pair(apply_k, init, grad_weight_k, self.chunk)
        self.lora_A_v, self.lora_B_v = make_pair(apply_v, init, grad_weight_v, 2 * self.chunk)

    def _delta(self, x, A, B):
        if A is None or B is None:
            return None
        return ((self.dropout(x) @ A.T) @ B.T) * self.scaling # mathematically correct

    def forward(self, x):
        out = self.base_layer(x)
        if not self.has_any_adapter:
            return out
        chunks = list(out.split(self.chunk, dim=-1))
        delta_q = self._delta(x, self.lora_A_q, self.lora_B_q)
        delta_k = self._delta(x, self.lora_A_k, self.lora_B_k)
        delta_v = self._delta(x, self.lora_A_v, self.lora_B_v)
        if delta_q is not None:
            chunks[0] = chunks[0] + delta_q
        if delta_k is not None:
            chunks[1] = chunks[1] + delta_k
        if delta_v is not None:
            chunks[2] = chunks[2] + delta_v
        return torch.cat(chunks, dim=-1)

    def shared_parameters(self):
        params = []
        for p in [self.lora_A_q, self.lora_B_q, self.lora_A_k, self.lora_B_k, self.lora_A_v, self.lora_B_v]:
            if p is not None:
                params.append(p)
        return params


class TimmLoRAAdapter(nn.Module):
    """Wrapper around a ViT model patched with LoRA modules."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def shared_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class TimmVPTAdapter(nn.Module):
    """Visual Prompt Tuning wrapper for timm ViT models."""

    def __init__(
        self,
        model,
        num_tokens=5,
        prompt_dropout=0.0,
        prompt_proj_dim=-1,
        deep=False,
        num_deep_layers=None,
        prompt_mlp_layers=0,
    ):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)
        self.deep = bool(deep)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))

        for p in self.model.parameters():
            p.requires_grad = False
        if not _unfreeze_classifier_head(self.model):
            log.warning("VPT PEFT requested but no classifier head was found to unfreeze.")

        embed_dim = int(getattr(model, "embed_dim", getattr(model, "num_features")))
        prompt_dim = int(prompt_proj_dim) if prompt_proj_dim is not None and int(prompt_proj_dim) > 0 else embed_dim

        patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", (16, 16))
        if isinstance(patch_size, int):
            patch_area = patch_size * patch_size
        else:
            patch_area = int(patch_size[0]) * int(patch_size[1])
        val = math.sqrt(6.0 / float(3 * patch_area + prompt_dim))

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        self.prompt_proj = nn.Linear(prompt_dim, embed_dim) if prompt_dim != embed_dim else nn.Identity()

        mlp_layers = []
        for _ in range(int(prompt_mlp_layers)):
            mlp_layers.append(nn.Linear(embed_dim, embed_dim))
            mlp_layers.append(nn.GELU())
        self.prompt_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.prompt_dropout = nn.Dropout(float(prompt_dropout)) if prompt_dropout and prompt_dropout > 0 else nn.Identity()

        if self.deep:
            total_layers = len(self.model.blocks) - 1
            self.num_deep_layers = int(num_deep_layers) if num_deep_layers is not None else total_layers
            self.num_deep_layers = max(0, min(self.num_deep_layers, total_layers))
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(self.num_deep_layers, self.num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings, -val, val)
        else:
            self.deep_prompt_embeddings = None

    def _transform_prompt(self, prompt_emb):
        prompt = self.prompt_proj(prompt_emb)
        prompt = self.prompt_mlp(prompt)
        prompt = self.prompt_dropout(prompt)
        return prompt

    def _insert_prompts(self, x, prompt_tokens):
        return torch.cat(
            [
                x[:, : self.num_prefix_tokens, :],
                prompt_tokens,
                x[:, self.num_prefix_tokens :, :],
            ],
            dim=1,
        )

    def _strip_prompts(self, x):
        return torch.cat(
            [
                x[:, : self.num_prefix_tokens, :],
                x[:, self.num_prefix_tokens + self.num_tokens :, :],
            ],
            dim=1,
        )

    def _pos_embed(self, x):
        if hasattr(self.model, "_pos_embed"):
            return self.model._pos_embed(x)

        B = x.shape[0]
        prefix = []
        if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
            prefix.append(self.model.cls_token.expand(B, -1, -1))
        if hasattr(self.model, "reg_token") and self.model.reg_token is not None:
            prefix.append(self.model.reg_token.expand(B, -1, -1))
        if len(prefix) > 0:
            x = torch.cat([*prefix, x], dim=1)
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            if self.model.pos_embed.shape[1] == x.shape[1]:
                x = x + self.model.pos_embed
        if hasattr(self.model, "pos_drop"):
            x = self.model.pos_drop(x)
        return x

    def _forward_features(self, x):
        x = self.model.patch_embed(x)
        x = self._pos_embed(x)

        if hasattr(self.model, "patch_drop"):
            x = self.model.patch_drop(x)
        if hasattr(self.model, "norm_pre"):
            x = self.model.norm_pre(x)

        B = x.shape[0]
        shallow_prompt = self._transform_prompt(self.prompt_embeddings).expand(B, -1, -1)
        x = self._insert_prompts(x, shallow_prompt)

        for idx, block in enumerate(self.model.blocks):
            if self.deep_prompt_embeddings is not None and idx > 0 and (idx - 1) < self.deep_prompt_embeddings.shape[0]:
                deep_prompt = self._transform_prompt(self.deep_prompt_embeddings[idx - 1]).expand(B, -1, -1)
                x = self._insert_prompts(self._strip_prompts(x), deep_prompt)
            x = block(x)

        x = self._strip_prompts(x)
        if hasattr(self.model, "norm"):
            x = self.model.norm(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        if hasattr(self.model, "forward_head"):
            return self.model.forward_head(x)
        if x.ndim == 3:
            if getattr(self.model, "global_pool", "") == "avg":
                x = x[:, self.num_prefix_tokens :, :].mean(dim=1)
            else:
                x = x[:, 0]
        if hasattr(self.model, "fc_norm") and self.model.fc_norm is not None:
            x = self.model.fc_norm(x)
        if hasattr(self.model, "head_drop"):
            x = self.model.head_drop(x)
        if hasattr(self.model, "head"):
            x = self.model.head(x)
        return x

    def shared_parameters(self):
        params = [self.prompt_embeddings]
        params += list(self.prompt_proj.parameters()) if isinstance(self.prompt_proj, nn.Module) else []
        params += list(self.prompt_mlp.parameters()) if isinstance(self.prompt_mlp, nn.Module) else []
        if self.deep_prompt_embeddings is not None:
            params.append(self.deep_prompt_embeddings)
        # Include any trainable base model params (e.g., classification head).
        params += [p for p in self.model.parameters() if p.requires_grad]
        return params


class CNNPromptTuningAdapter(nn.Module):
    """
    CNN Prompt Tuning using learnable padding.
    
    Adds trainable border pixels around the input image.
    The backbone CNN is frozen.
    """

    def __init__(self, model, image_size, prompt_size=4):
        """
        Args:
            model: pretrained CNN
            image_size: (C, H, W)
            prompt_size: number of pixels to pad on each side
        """
        super().__init__()

        if image_size is None or len(image_size) != 3:
            raise ValueError("image_size must be a 3-tuple/list (C, H, W).")
        if int(prompt_size) < 0:
            raise ValueError("prompt_size must be non-negative.")

        self.model = model
        self.prompt_size = int(prompt_size)

        C, H, W = [int(v) for v in image_size]
        if C <= 0 or H <= 0 or W <= 0:
            raise ValueError("image_size entries must be positive.")
        self.C = C
        self.H = H
        self.W = W

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False
        # Keep stateful layers (e.g., BatchNorm) fixed during adapter training.
        self.model.eval()

        # Learnable border prompt over padded canvas.
        self.prompt = nn.Parameter(
            torch.zeros(1, C, H + 2 * self.prompt_size, W + 2 * self.prompt_size)
        )
        mask = torch.ones_like(self.prompt)
        p = self.prompt_size
        if p > 0:
            mask[:, :, p : p + H, p : p + W] = 0.0
        self.register_buffer("prompt_mask", mask)

        nn.init.uniform_(self.prompt, -0.02, 0.02)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        if x.ndim != 4:
            raise ValueError("Input x must have shape (B, C, H, W).")
        if x.shape[1] != self.C or x.shape[2] != self.H or x.shape[3] != self.W:
            raise ValueError(
                f"Input shape mismatch: expected (B, {self.C}, {self.H}, {self.W}), got {tuple(x.shape)}."
            )

        B = x.shape[0]
        p = self.prompt_size

        # Border-only prompt: center region is always zeroed and replaced by input.
        prompt = (self.prompt * self.prompt_mask).to(dtype=x.dtype)
        prompt = prompt.expand(B, -1, -1, -1)

        # Insert image into center (no additive prompt on image content).
        padded = prompt.clone()
        padded[:, :, p : p + self.H, p : p + self.W] = x

        return self.model(padded)

    def shared_parameters(self):
        return [self.prompt]

def _is_vit_backbone(model):
    return hasattr(model, "patch_embed") and hasattr(model, "blocks") and hasattr(model, "forward")


def _find_vit_backbone(model):
    if _is_vit_backbone(model):
        return model, None
    if hasattr(model, "model") and _is_vit_backbone(model.model):
        return model.model, model
    return None, None


def _parse_peft_cfg(cfg_model, kwargs):
    peft_cfg = _cfg_get(cfg_model, "peft", None) if not isinstance(cfg_model, str) else None
    if peft_cfg is None and "cfg_case" in kwargs:
        peft_cfg = _cfg_get(kwargs["cfg_case"], "peft", None)
    return peft_cfg


def _replace_linear_with_lora(
    module_parent,
    attr_name,
    rank,
    alpha,
    dropout,
    init,
    grad_weight=None,
    lora_one_gamma=2.0,
    use_rslora=False,
):
    old_layer = getattr(module_parent, attr_name, None)
    if isinstance(old_layer, nn.Linear):
        setattr(
            module_parent,
            attr_name,
            LoRALinear(
                old_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                init=init,
                grad_weight=grad_weight,
                lora_one_gamma=lora_one_gamma,
                use_rslora=use_rslora,
            ),
        )
        return 1
    return 0


def _unfreeze_classifier_head(model):
    head_modules = []
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            head_modules.append(classifier)
        elif isinstance(classifier, str):
            module = getattr(model, classifier, None)
            if isinstance(module, nn.Module):
                head_modules.append(module)
    for attr in ("head", "head_dist", "classifier", "fc", "last_linear"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            head_modules.append(module)

    found = False
    for module in set(head_modules):
        for p in module.parameters():
            p.requires_grad = True
            found = True
    return found


def _apply_lora_to_vit(backbone, peft_cfg, cfg_case=None):
    lora_cfg = _cfg_get(peft_cfg, "lora", peft_cfg)
    rank = int(_cfg_get(lora_cfg, "rank", 4))
    alpha = float(_cfg_get(lora_cfg, "alpha", rank))
    dropout = float(_cfg_get(lora_cfg, "dropout", 0.0))
    init = _normalize_lora_init(_cfg_get(lora_cfg, "initialisation", "standard"))
    lora_one_gamma = float(_cfg_get(lora_cfg, "gamma", _default_lora_one_gamma(cfg_case)))
    use_rslora = bool(_cfg_get(lora_cfg, "use_rslora", _is_lora_one_init(init)))
    log.info("Using LoRA initialization strategy: %s.", init)
    targets = _cfg_get(lora_cfg, "target_modules", _cfg_get(lora_cfg, "targets", ["q", "k", "v"]))
    if isinstance(targets, str):
        targets = [target.strip() for target in targets.split(",")]
    target_set = {target.lower() for target in targets}
    grad_map = {}
    if _is_lora_one_init(init):
        grad_map = _compute_lora_one_weight_grads(backbone, peft_cfg, cfg_case, target_set)

    for p in backbone.parameters():
        p.requires_grad = False

    adapters = 0
    for block in backbone.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                qkv_grad = grad_map.get(id(attn.qkv.weight), None)
                q_grad, k_grad, v_grad = None, None, None
                if qkv_grad is not None and qkv_grad.ndim == 2 and qkv_grad.shape[0] % 3 == 0:
                    chunk = qkv_grad.shape[0] // 3
                    q_grad, k_grad, v_grad = qkv_grad[:chunk], qkv_grad[chunk : 2 * chunk], qkv_grad[2 * chunk :]
                qkv_layer = LoRAQKVLinear(
                    attn.qkv,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    apply_q=("q" in target_set),
                    apply_k=("k" in target_set),
                    apply_v=("v" in target_set),
                    init=init,
                    grad_weight_q=q_grad,
                    grad_weight_k=k_grad,
                    grad_weight_v=v_grad,
                    lora_one_gamma=lora_one_gamma,
                    use_rslora=use_rslora,
                )
                if qkv_layer.has_any_adapter:
                    attn.qkv = qkv_layer
                    adapters += 1
            else:
                if "q" in target_set:
                    grad_weight = grad_map.get(id(attn.q_proj.weight), None) if hasattr(attn, "q_proj") else None
                    adapters += _replace_linear_with_lora(
                        attn,
                        "q_proj",
                        rank,
                        alpha,
                        dropout,
                        init,
                        grad_weight=grad_weight,
                        lora_one_gamma=lora_one_gamma,
                        use_rslora=use_rslora,
                    )
                if "k" in target_set:
                    grad_weight = grad_map.get(id(attn.k_proj.weight), None) if hasattr(attn, "k_proj") else None
                    adapters += _replace_linear_with_lora(
                        attn,
                        "k_proj",
                        rank,
                        alpha,
                        dropout,
                        init,
                        grad_weight=grad_weight,
                        lora_one_gamma=lora_one_gamma,
                        use_rslora=use_rslora,
                    )
                if "v" in target_set:
                    grad_weight = grad_map.get(id(attn.v_proj.weight), None) if hasattr(attn, "v_proj") else None
                    adapters += _replace_linear_with_lora(
                        attn,
                        "v_proj",
                        rank,
                        alpha,
                        dropout,
                        init,
                        grad_weight=grad_weight,
                        lora_one_gamma=lora_one_gamma,
                        use_rslora=use_rslora,
                    )

            if "proj" in target_set:
                grad_weight = grad_map.get(id(attn.proj.weight), None) if hasattr(attn, "proj") else None
                adapters += _replace_linear_with_lora(
                    attn,
                    "proj",
                    rank,
                    alpha,
                    dropout,
                    init,
                    grad_weight=grad_weight,
                    lora_one_gamma=lora_one_gamma,
                    use_rslora=use_rslora,
                )

        if "mlp" in target_set:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                grad_fc1 = grad_map.get(id(mlp.fc1.weight), None) if hasattr(mlp, "fc1") else None
                grad_fc2 = grad_map.get(id(mlp.fc2.weight), None) if hasattr(mlp, "fc2") else None
                adapters += _replace_linear_with_lora(
                    mlp,
                    "fc1",
                    rank,
                    alpha,
                    dropout,
                    init,
                    grad_weight=grad_fc1,
                    lora_one_gamma=lora_one_gamma,
                    use_rslora=use_rslora,
                )
                adapters += _replace_linear_with_lora(
                    mlp,
                    "fc2",
                    rank,
                    alpha,
                    dropout,
                    init,
                    grad_weight=grad_fc2,
                    lora_one_gamma=lora_one_gamma,
                    use_rslora=use_rslora,
                )

    if adapters == 0:
        log.warning("LoRA PEFT requested but no target layers were patched.")
    if not _unfreeze_classifier_head(backbone):
        log.warning("LoRA PEFT requested but no classifier head was found to unfreeze.")
    return TimmLoRAAdapter(backbone)


def _apply_vpt_to_vit(backbone, peft_cfg):
    vpt_cfg = _cfg_get(peft_cfg, "vpt", peft_cfg)
    return TimmVPTAdapter(
        backbone,
        num_tokens=int(_cfg_get(vpt_cfg, "num_tokens", 5)),
        prompt_dropout=float(_cfg_get(vpt_cfg, "dropout", 0.0)),
        prompt_proj_dim=int(_cfg_get(vpt_cfg, "project", -1)),
        deep=bool(_cfg_get(vpt_cfg, "deep", False)),
        num_deep_layers=_cfg_get(vpt_cfg, "num_deep_layers", None),
        prompt_mlp_layers=int(_cfg_get(vpt_cfg, "prompt_mlp_layers", _cfg_get(vpt_cfg, "prompt_mlp_num", 0))),
    )


def _apply_cnn_prompt_tuning(model, peft_cfg, cfg_data=None):
    cnn_cfg = _cfg_get(peft_cfg, "cnn_prompt", peft_cfg)
    prompt_size = int(_cfg_get(cnn_cfg, "prompt_size", _cfg_get(cnn_cfg, "size", 4)))
    image_size = _cfg_get(cnn_cfg, "image_size", None)
    if image_size is None and cfg_data is not None and hasattr(cfg_data, "shape"):
        image_size = tuple(int(v) for v in cfg_data.shape[:3])
    if image_size is None:
        raise ValueError("CNN prompt PEFT requires `image_size` or available `cfg_data.shape`.")
    if len(image_size) != 3:
        raise ValueError("CNN prompt `image_size` must be a 3-tuple/list (C, H, W).")
    return CNNPromptTuningAdapter(model, image_size=image_size, prompt_size=prompt_size)


def _adapt_model_with_peft(model, cfg_model, modality="vision", **kwargs):
    if modality != "vision":
        return model
    model_name = _model_name(cfg_model).lower()

    peft_cfg = _parse_peft_cfg(cfg_model, kwargs)
    if peft_cfg is None or not bool(_cfg_get(peft_cfg, "enabled", True)):
        return model

    peft_type = _cfg_get(peft_cfg, "type", None)
    if peft_type is None:
        if _cfg_get(peft_cfg, "vpt", None) is not None:
            peft_type = "vpt"
        elif _cfg_get(peft_cfg, "lora", None) is not None:
            peft_type = "lora"
        elif _cfg_get(peft_cfg, "cnn_prompt", None) is not None:
            peft_type = "cnn_prompt"
    if peft_type is None:
        raise ValueError("PEFT config requires `type` to be one of `vpt`, `lora`, or `cnn_prompt`.")

    peft_type = str(peft_type).lower()
    if peft_type in ("cnn_prompt", "cnn-prompt"):
        log.info("Applying %s PEFT adaptation to CNN model %s.", peft_type, model_name)
        return _apply_cnn_prompt_tuning(model, peft_cfg, cfg_data=_cfg_get(kwargs, "cfg_data", None))

    backbone, parent = _find_vit_backbone(model)
    if backbone is None:
        if peft_type in ("vpt", "lora"):
            log.warning("PEFT requested for %s but no compatible ViT backbone was found.", model_name)
        return model

    log.info("Applying %s PEFT adaptation to ViT model %s.", peft_type, model_name)
    if peft_type == "vpt":
        adapted_backbone = _apply_vpt_to_vit(backbone, peft_cfg)
    elif peft_type == "lora":
        adapted_backbone = _apply_lora_to_vit(backbone, peft_cfg, cfg_case=_cfg_get(kwargs, "cfg_case", None))
    else:
        raise ValueError(f"Unknown PEFT type {peft_type}.")
    
    # Can set the positional embedding to be trainable or not (default is not trainable for ViT PEFT):
    posembed_train = _cfg_get(cfg_model, "posembed_trainable", None)
    log.info("Setting ViT positional embedding requires_grad to %s.", posembed_train)
    if posembed_train is not None:
        # log.info(f"posembed is set to {adapted_backbone.model.pos_embed.requires_grad}.")
        adapted_backbone.model.pos_embed.requires_grad_(posembed_train)
        # log.info(f"posembed is set to {adapted_backbone.model.pos_embed.requires_grad}.")
    #################################################################################################
    if parent is None:
        return adapted_backbone
    parent.model = adapted_backbone
    return model


class BlockWithPos(nn.Module):
    """Wrapper for ViT block with learnable positional embedding."""

    def __init__(self, block, dim, num_tokens, base_pos_embed=None):
        super().__init__()
        self.block = block

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))

        # Initialize from pretrained positional embedding (important)
        if base_pos_embed is not None:
            with torch.no_grad():
                self.pos_embed.copy_(base_pos_embed)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Safer: start from 0 (no extra positional effect initially)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x + self.gamma * self.pos_embed
        return self.block(x)


def _apply_block_wise_positional_embeddings(model):
    """Apply block-wise positional embeddings starting from block 2."""

    backbone, _ = _find_vit_backbone(model)
    if backbone is None:
        return model

    base_pos_embed = backbone.pos_embed
    num_tokens = base_pos_embed.shape[1]
    dim = base_pos_embed.shape[2]

    new_blocks = []

    for idx, block in enumerate(backbone.blocks):

        if idx == 0:
            # First block → NO extra positional embedding
            new_blocks.append(block)

        else:
            wrapped_block = BlockWithPos(
                block,
                dim,
                num_tokens,
                base_pos_embed=base_pos_embed
            )
            new_blocks.append(wrapped_block)

    backbone.blocks = nn.Sequential(*new_blocks)

    # Keep original positional embedding trainable
    backbone.pos_embed.requires_grad_(True)

    # log.info("Applied block-wise positional embeddings (from block 2 onward).")
    # for i, blk in enumerate(backbone.blocks):
    #     if hasattr(blk, "pos_embed"):
    #         print(f"Block {i}: has pos_embed")
    #     else:
    #         print(f"Block {i}: NO pos_embed")

    return model


def construct_model(cfg_model, cfg_data, pretrained=True, pretrained_path=None, **kwargs):
    if cfg_data.modality == "vision":
        model = _construct_vision_model(cfg_model, cfg_data, pretrained, pretrained_path=pretrained_path, **kwargs)
    elif cfg_data.modality == "text":
        model = _construct_text_model(cfg_model, cfg_data, pretrained, pretrained_path=pretrained_path, **kwargs)
    else:
        raise ValueError(f"Invalid data modality {cfg_data.modality}")

    # Save nametag for printouts later:
    model.name = _model_name(cfg_model)

    # Choose loss function according to data and model:
    if "classification" in cfg_data.task:
        loss_fn = torch.nn.CrossEntropyLoss()
    elif "causal-lm-sanity" in cfg_data.task:
        loss_fn = MostlyCausalLoss()
    elif "causal-lm" in cfg_data.task:
        loss_fn = CausalLoss()
    elif "masked-lm" in cfg_data.task:
        loss_fn = MLMLoss(vocab_size=cfg_data.vocab_size)
    else:
        raise ValueError(f"No loss function registered for task {cfg_data.task}.")
    loss_fn = torch.jit.script(loss_fn)
    return model, loss_fn


def _construct_text_model(cfg_model, cfg_data, pretrained=True, pretrained_path=None, **kwargs):
    raw_cfg_model = cfg_model
    cfg_model = _model_name(cfg_model)
    loaded_from_checkpoint = False
    if cfg_model == "transformer3f":
        # This is the transformer from "A field guide to federated learning"
        """
        we train a modified 3-layer Transformer model [250],
        where the dimension of the token embeddings is 96, and the hidden dimension of the feed-forward
        network (FFN) block is 1536. We use 8 heads for the multi-head attention, where each head is based
        on 12-dimensional (query, key, value) vectors. We use ReLU activation and set dropout rate to 0.1.
        """
        # For simplicity the dropout is disabled for now
        # the 12-dim query is 96/8 = 12
        model = TransformerModel(
            ntokens=cfg_data.vocab_size, ninp=96, nhead=8, nhid=1536, nlayers=3, dropout=0, positional_embedding="fixed"
        )
    elif cfg_model == "transformer3":
        # Same as above but with learnable positional embeddings
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
        )
    elif cfg_model == "transformer3t":
        # Same as above but with learnable positional embeddings and tied weights
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
            tie_weights=True,
        )
    elif cfg_model == "transformer1":
        # This is our initial sanity check transformer:
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=200, nhead=1, nhid=200, nlayers=1, dropout=0)
    elif cfg_model == "transformerS":
        # A wide sanity-check transformer
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=512, nhead=1, nhid=512, nlayers=1, dropout=0)
    elif cfg_model == "LSTM":
        # This is the LSTM from "LEARNING DIFFERENTIALLY PRIVATE RECURRENT LANGUAGE MODELS"
        """
        word s t is mapped to an embedding vector e t ∈ R 96
        by looking up the word in the model’s vocabulary. The e t is composed with the state emitted by
        the model in the previous time step s t−1 ∈ R 256 to emit a new state vector s t and an “output
        embedding” o t ∈ R 96 .
        """
        model = RNNModel("LSTM", cfg_data.vocab_size, ninp=96, nhid=96, nlayers=1, dropout=0.0, tie_weights=True)
    elif cfg_model == "linear":
        model = LinearModel(cfg_data.vocab_size, embedding_size=200)
    else:
        try:
            from transformers import (
                AutoModelForMaskedLM,
                AutoModelForPreTraining,
                AutoModelForSequenceClassification,
                AutoConfig,
            )

            if cfg_data.task == "masked-lm":
                auto_class = AutoModelForMaskedLM
            elif cfg_data.task == "classification":
                auto_class = AutoModelForSequenceClassification
            else:
                auto_class = AutoModelForPreTraining
            # Make sure to use the matching tokenizer and vocab_size!
            if cfg_model == "gpt2S":
                cfg_model = "gpt2"
                extra_args = dict(activation_function="relu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
            elif cfg_model == "bert-sanity-check":
                cfg_model = "bert-base-uncased"
                extra_args = dict(hidden_act="relu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            else:
                extra_args = dict()
            if pretrained_path is not None:
                pretrained_path = Path(pretrained_path).expanduser()
                if pretrained_path.is_dir():
                    model = auto_class.from_pretrained(str(pretrained_path), **extra_args)
                    loaded_from_checkpoint = True
                else:
                    hf_cfg = AutoConfig.from_pretrained(cfg_model, **extra_args)
                    model = auto_class.from_config(hf_cfg)
                    _load_pretrained_weights(model, pretrained_path)
                    loaded_from_checkpoint = True
            elif pretrained:
                model = auto_class.from_pretrained(cfg_model, **extra_args)
            else:
                hf_cfg = AutoConfig.from_pretrained(cfg_model, **extra_args)
                model = auto_class.from_config(hf_cfg)
            # model.transformer.h[0].attn.scale_attn_weights = False
            if model.config.vocab_size != cfg_data.vocab_size:
                model.resize_token_embeddings(new_num_tokens=cfg_data.vocab_size)
            model = HuggingFaceContainer(model)
        except OSError as error_msg:
            raise ValueError(f"Invalid huggingface model {cfg_model} given: {error_msg}")
    if pretrained_path is not None and not loaded_from_checkpoint:
        _load_pretrained_weights(model, pretrained_path)
    return _adapt_model_with_peft(model, raw_cfg_model, modality="text", **kwargs)


class HuggingFaceContainer(torch.nn.Module):
    """Wrap huggingface models for a unified interface. Ugh."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if "inputs" in kwargs:
            kwargs["input_ids"] = kwargs.pop("inputs")
        if "input_ids" not in kwargs:
            kwargs["input_ids"] = args[0]
        if kwargs["input_ids"].dtype != torch.long:
            kwargs["inputs_embeds"] = kwargs.pop("input_ids")
        outputs = self.model(**kwargs)
        return outputs["logits"] if "logits" in outputs else outputs["prediction_logits"]

    def shared_parameters(self):
        return self.model.parameters()


class VisionContainer(torch.nn.Module):
    """We'll use a container to catch extra attributes and allow for usage with model(**data)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        return self.model(inputs)

    def shared_parameters(self):
        if hasattr(self.model, "shared_parameters"):
            return self.model.shared_parameters()
        return self.model.parameters()


def _construct_vision_model(cfg_model, cfg_data, pretrained=True, pretrained_path=None, **kwargs):
    """Construct the neural net that is used."""
    raw_cfg_model = cfg_model
    cfg_model = _model_name(cfg_model)
    channels = cfg_data.shape[0]
    classes = cfg_data.classes
    use_pretrained_path = pretrained_path is not None
    loaded_from_checkpoint = False

    if "ImageNet" in cfg_data.name:
        try:
            model = getattr(torchvision.models, cfg_model.lower())(pretrained=False if use_pretrained_path else pretrained)
            try:
                # Try to adjust the linear layer and fill with previous data
                fc = torch.nn.Linear(model.fc.in_features, classes)
                if pretrained and not use_pretrained_path:
                    fc.weight.data = model.fc.weight[:classes]
                    fc.bias.data = model.fc.bias[:classes]
                model.fc = fc
            except AttributeError:
                pass
            _load_pretrained_weights(model, pretrained_path)
            loaded_from_checkpoint = use_pretrained_path
        except AttributeError:
            if "nfnet" in cfg_model:
                model = NFNet(
                    channels,
                    classes,
                    variant="F0",
                    stochdepth_rate=0.25,
                    alpha=0.2,
                    se_ratio=0.5,
                    activation="ReLU",
                    stem="ImageNet",
                    use_dropout=True,
                )
                if pretrained_path is not None:
                    _load_pretrained_weights(model, pretrained_path)
                    loaded_from_checkpoint = True
            elif "resnet101wsl" in cfg_model:
                model = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
                if pretrained_path is not None:
                    _load_pretrained_weights(model, pretrained_path)
                    loaded_from_checkpoint = True
            elif "resnet50swsl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl")
                if pretrained_path is not None:
                    _load_pretrained_weights(model, pretrained_path)
                    loaded_from_checkpoint = True
            elif "resnet50ssl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl")
                if pretrained_path is not None:
                    _load_pretrained_weights(model, pretrained_path)
                    loaded_from_checkpoint = True
            elif "resnetmoco" in cfg_model:
                model = torchvision.models.resnet50(pretrained=False)
                if pretrained and not use_pretrained_path:
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
                    url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar"
                    state_dict = torch.hub.load_state_dict_from_url(
                        url, progress=True, map_location=torch.device("cpu")
                    )["state_dict"]
                    for key in list(state_dict.keys()):
                        val = state_dict.pop(key)
                        # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                        sanitized_key = key.replace("module.", "")
                        state_dict[sanitized_key] = val

                    model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
                    loaded_from_checkpoint = True
            elif "vit_base_april" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=False if use_pretrained_path else pretrained)
                _load_pretrained_weights(model, pretrained_path)
                loaded_from_checkpoint = use_pretrained_path
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_small_april" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=False if use_pretrained_path else pretrained)
                _load_pretrained_weights(model, pretrained_path)
                loaded_from_checkpoint = use_pretrained_path
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_base" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=False if use_pretrained_path else pretrained)
                _load_pretrained_weights(model, pretrained_path)
                loaded_from_checkpoint = use_pretrained_path
            elif "vit_small" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=False if use_pretrained_path else pretrained)
                _load_pretrained_weights(model, pretrained_path)
                loaded_from_checkpoint = use_pretrained_path

            elif "linear" == cfg_model:
                input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
            elif "none" == cfg_model:
                model = torch.nn.Sequential(torch.nn.Flatten(), _Select(classes))
            else:
                raise ValueError(f"Could not find ImageNet model {cfg_model} in torchvision.models or custom models.")
            if pretrained_path is not None and not loaded_from_checkpoint:
                _load_pretrained_weights(model, pretrained_path)
    else:
        # CIFAR Model from here:
        if "resnetgn" in cfg_model.lower():
            block, layers = resnet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="groupnorm4th",
                downsample="B",
                width_per_group=16 if len(layers) < 4 else 64,
                zero_init_residual=False,
            )
        elif "resnet" in cfg_model.lower():
            if "-" in cfg_model.lower():  # Hacky way to separate ResNets from wide ResNets which are e.g. 28-10
                depth = int("".join(filter(str.isdigit, cfg_model.split("-")[0])))
                width = int("".join(filter(str.isdigit, cfg_model.split("-")[1])))
            else:
                depth = int("".join(filter(str.isdigit, cfg_model)))
                width = 1
            block, layers = resnet_depths_to_config(depth)
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="BatchNorm2d",
                downsample="B",
                width_per_group=(16 if len(layers) < 4 else 64) * width,
                zero_init_residual=False,
            )
        elif "densenet" in cfg_model.lower():
            growth_rate, block_config, num_init_features = densenet_depths_to_config(
                int("".join(filter(str.isdigit, cfg_model)))
            )
            model = DenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_init_features=num_init_features,
                bn_size=4,
                drop_rate=0,
                channels=channels,
                num_classes=classes,
                memory_efficient=False,
                norm="BatchNorm2d",
                nonlin="ReLU",
                stem="CIFAR",
                convolution_type="Standard",
            )
        elif "vgg" in cfg_model.lower():
            model = VGG(
                cfg_model,
                in_channels=channels,
                num_classes=classes,
                norm="BatchNorm2d",
                nonlin="ReLU",
                head="CIFAR",
                convolution_type="Standard",
                drop_rate=0,
                classical_weight_init=True,
            )
        elif "linear" in cfg_model:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
        elif "nfnet" in cfg_model:
            model = NFNet(
                channels,
                classes,
                variant="F0",
                stochdepth_rate=0.25,
                alpha=0.2,
                se_ratio=0.5,
                activation="ReLU",
                stem="CIFAR",
                use_dropout=True,
            )
        elif "convnet-trivial" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv", torch.nn.Conv2d(channels, 3072, 3, stride=1, padding=1)),
                        ("relu", torch.nn.ReLU(inplace=True)),
                        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
                        ("flatten", torch.nn.Flatten()),
                        ("linear", torch.nn.Linear(3072, classes)),
                    ]
                )
            )
        elif "convnetsmall" == cfg_model.lower():
            model = ConvNetSmall(width=256, num_channels=channels, num_classes=classes)
        elif "convnet" == cfg_model.lower():
            model = ConvNet(width=64, num_channels=channels, num_classes=classes)
        elif "convnet_beyond" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", torch.nn.Conv2d(channels, 32, 3, stride=2, padding=1)),
                        ("relu0", torch.nn.LeakyReLU()),
                        ("conv2", torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                        ("relu1", torch.nn.LeakyReLU()),
                        ("conv3", torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                        ("relu2", torch.nn.LeakyReLU()),
                        ("conv4", torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                        ("relu3", torch.nn.LeakyReLU()),
                        ("flatt", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(12544, 12544)),
                        ("relu4", torch.nn.LeakyReLU()),
                        ("linear1", torch.nn.Linear(12544, classes)),
                        ("softmax", torch.nn.Softmax(dim=1)),
                    ]
                )
            )
        elif "lenet_zhu" == cfg_model.lower():
            model = LeNetZhu(num_channels=channels, num_classes=classes)
        elif "cnn6" == cfg_model.lower():
            # This is the model from R-GAP:
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("layer0", torch.nn.Conv2d(channels, 12, kernel_size=4, padding=2, stride=2, bias=False)),
                        ("act0", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer1", torch.nn.Conv2d(12, 36, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act1", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer2", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act2", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer3", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act3", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer4", torch.nn.Conv2d(36, 64, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act4", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer5", torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("flatten", torch.nn.Flatten()),
                        ("act5", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("fc", torch.nn.Linear(3200, classes, bias=True)),
                    ]
                )
            )
        elif cfg_model == "MLP":
            width = 1024
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(3072, width)),
                        ("relu0", torch.nn.ReLU()),
                        ("linear1", torch.nn.Linear(width, width)),
                        ("relu1", torch.nn.ReLU()),
                        ("linear2", torch.nn.Linear(width, width)),
                        ("relu2", torch.nn.ReLU()),
                        ("linear3", torch.nn.Linear(width, classes)),
                    ]
                )
            )
        elif "vit_base" in cfg_model:
            import timm
            img_size = _resolve_vit_image_size(cfg_model, cfg_data)
            if img_size is None:
                img_size = (cfg_data.shape[1], cfg_data.shape[2])
            in_chans = cfg_data.shape[0]
            p = _resolve_vit_patch_size(raw_cfg_model, cfg_data, default_patch_size=14 if "MNIST" in cfg_data.name else 16)
            embed_layer = _cfg_get(raw_cfg_model, "embed_layer", "conv")
            if embed_layer == 'conv':
                model = timm.create_model("vit_base_patch16_224", pretrained=False if p != 16 or use_pretrained_path else pretrained,
                                            img_size=img_size, 
                                            in_chans=in_chans, 
                                            num_classes=classes,
                                            patch_size=p)
            elif embed_layer == 'linear':
                model = timm.create_model("vit_base_patch16_224", pretrained=False if p != 16 or use_pretrained_path else pretrained,
                                            img_size=img_size, 
                                            in_chans=in_chans, 
                                            num_classes=classes,
                                            patch_size=p,
                                            embed_layer=LinearPatchEmbed)
            if use_pretrained_path:
                _load_pretrained_weights(model, pretrained_path)
            loaded_from_checkpoint = use_pretrained_path
            if "april" in cfg_model:
                model.model.blocks[0] = ModifiedBlock(model.model.blocks[0])
        elif "vit_small" in cfg_model:
            import timm
            img_size = _resolve_vit_image_size(cfg_model, cfg_data)
            if img_size is None:
                img_size = (cfg_data.shape[1], cfg_data.shape[2])
            in_chans = cfg_data.shape[0]
            p = _resolve_vit_patch_size(raw_cfg_model, cfg_data, default_patch_size=14 if "MNIST" in cfg_data.name else 16)
            embed_layer = _cfg_get(raw_cfg_model, "embed_layer", "conv")
            if embed_layer == 'conv':
                model = timm.create_model("vit_small_patch16_224", pretrained=False if p != 16 or use_pretrained_path else pretrained,
                                            img_size=img_size, 
                                            in_chans=in_chans, 
                                            num_classes=classes,
                                            patch_size=p)
            elif embed_layer == 'linear':
                model = timm.create_model("vit_small_patch16_224", pretrained=False if p != 16 or use_pretrained_path else pretrained,
                                            img_size=img_size, 
                                            in_chans=in_chans, 
                                            num_classes=classes,
                                            patch_size=p,
                                            embed_layer=LinearPatchEmbed)
            if use_pretrained_path:
                _load_pretrained_weights(model, pretrained_path)
            loaded_from_checkpoint = use_pretrained_path
            if "april" in cfg_model:
                model.model.blocks[0] = ModifiedBlock(model.model.blocks[0])
        elif "custom" in cfg_model:
            # Provision for custom ViT or other models
            if "vit" in cfg_model:
                img_size = _resolve_vit_image_size(cfg_model, cfg_data)
                if img_size is None:
                    img_size = (cfg_data.shape[1], cfg_data.shape[2])
                in_chans = cfg_data.shape[0]
                p = _resolve_vit_patch_size(raw_cfg_model, cfg_data, default_patch_size=14 if "MNIST" in cfg_data.name else 16)
                model = SmallViT(img_size=img_size, 
                                in_chans=in_chans, 
                                num_classes=classes, 
                                patch_size=p, 
                                embed_dim=_cfg_get(raw_cfg_model, "embed_dim", 384), 
                                depth=_cfg_get(raw_cfg_model, "depth", 7), 
                                num_heads=_cfg_get(raw_cfg_model, "num_heads", 12),
                                embed_layer=_cfg_get(raw_cfg_model, "embed_layer", "conv"),
                                mlp_ratio=_cfg_get(raw_cfg_model, "mlp_ratio", 4.0))
                if use_pretrained_path:
                    _load_pretrained_weights(model, pretrained_path)
                loaded_from_checkpoint = use_pretrained_path
                if "april" in cfg_model:
                    model.model.blocks[0] = ModifiedBlock(model.model.blocks[0])
            else:
                raise ValueError(f"Custom model type not recognized in {cfg_model}")
        else:
            raise ValueError("Model could not be found.")

        if pretrained_path is not None and not loaded_from_checkpoint:
            _load_pretrained_weights(model, pretrained_path)

    model = _adapt_model_with_peft(model, raw_cfg_model, modality="vision", cfg_data=cfg_data, **kwargs)

    # posembed_train = _cfg_get(raw_cfg_model, "posembed_trainable", None)
    # log.info("Setting ViT positional embedding requires_grad to %s.", posembed_train)
    # allow_block_pos_embed = _cfg_get(raw_cfg_model, "apply_block_pos_embed", None)
    # log.info("Allowing block-wise positional embeddings to be applied to ViT model: %s.", allow_block_pos_embed)
    if _cfg_get(raw_cfg_model, "apply_block_pos_embed", None):
        model = _apply_block_wise_positional_embeddings(model)

    return VisionContainer(model)


class ConvNetSmall(torch.nn.Module):
    """ConvNet without BN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu2", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv3", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu3", torch.nn.ReLU()),
                    ("pool1", torch.nn.AdaptiveAvgPool2d(1)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(4 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("bn0", torch.nn.BatchNorm2d(1 * width)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn1", torch.nn.BatchNorm2d(2 * width)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn2", torch.nn.BatchNorm2d(2 * width)),
                    ("relu2", torch.nn.ReLU()),
                    ("conv3", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn3", torch.nn.BatchNorm2d(4 * width)),
                    ("relu3", torch.nn.ReLU()),
                    ("conv4", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn4", torch.nn.BatchNorm2d(4 * width)),
                    ("relu4", torch.nn.ReLU()),
                    ("conv5", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn5", torch.nn.BatchNorm2d(4 * width)),
                    ("relu5", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv6", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn6", torch.nn.BatchNorm2d(4 * width)),
                    ("relu6", torch.nn.ReLU()),
                    ("conv7", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn7", torch.nn.BatchNorm2d(4 * width)),
                    ("relu7", torch.nn.ReLU()),
                    ("pool1", torch.nn.MaxPool2d(3)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(36 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(768, num_classes))
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class ModifiedBlock(torch.nn.Module):
    def __init__(self, old_Block):
        super().__init__()
        self.attn = old_Block.attn
        self.drop_path = old_Block.drop_path1
        self.norm2 = old_Block.norm2
        self.mlp = old_Block.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.drop_path(self.mlp((self.norm2(x))))
        return x


class SmallViT(torch.nn.Module):
    """A small ViT variant for low-resolution images like MNIST/CIFAR."""

    def __init__(self, img_size=(32, 32), 
                    patch_size=4, 
                    in_chans=3, 
                    num_classes=10, 
                    embed_dim=192, 
                    depth=4, 
                    num_heads=3, 
                    mlp_ratio=4.0,
                    embed_layer='conv'):
        super().__init__()
        import timm.models.vision_transformer
        if embed_layer=='conv':
            self.model = timm.models.vision_transformer.VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
        elif embed_layer=='linear':
            self.model = timm.models.vision_transformer.VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                embed_layer=LinearPatchEmbed
            )

    def forward(self, x):
        return self.model(x)

    def shared_parameters(self):
        if hasattr(self.model, "shared_parameters"):
            return self.model.shared_parameters()
        return self.model.parameters()
