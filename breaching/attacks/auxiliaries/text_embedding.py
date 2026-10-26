# Code for computing the similarity between text embeddings of class names from CIFAR-10 and ImageNet 
# using a specified text embedding model (e.g., SigLIP, CLIP, ALIGN). 
# This can be used to analyze how closely the class names from the two datasets are represented in the embedding space, 
# which may have implications for transfer learning and domain adaptation in the context of the GISMN attack.

################ To Prevent MUTEX Error ################
# The following environment variables are set to prevent potential MUTEX errors that can arise when using certain transformer models in a multi-threaded or multi-process environment.
# Setting "TRANSFORMERS_NO_TF" to "1" disables TensorFlow support in the Hugging Face Transformers library, which can help avoid conflicts if TensorFlow is not needed.
# Setting "USE_TF" to "0" explicitly tells the library not to use TensorFlow, which can further reduce the chances of encountering MUTEX errors related to TensorFlow's global state.
# Not required on Linux, but can be helpful on Windows or in certain environments where TensorFlow's threading model may cause issues.
import os
import sys

# On macOS/Windows, prevent transformers from loading TensorFlow/Keras to avoid
# pyarrow-related mutex crashes. Linux environments typically don't need this.
if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
########################################################

from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoTokenizer, AutoModel
from transformers import AlignModel, AlignTextModel
from transformers import SiglipTokenizer, SiglipTextModel
import torch
from .class_names import cifar10_classes, imagenet_classes
import torch.nn.functional as F

import logging

log = logging.getLogger(__name__)

def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    mean_col = gram.mean(dim=0, keepdim=True)
    mean_row = gram.mean(dim=1, keepdim=True)
    mean_all = gram.mean()
    return gram - mean_col - mean_row + mean_all

def centered_kernel_alignment(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute linear CKA between two embedding matrices of shape [n, d]."""
    if embeddings_a.ndim == 1:
        embeddings_a = embeddings_a.unsqueeze(0)
    if embeddings_b.ndim == 1:
        embeddings_b = embeddings_b.unsqueeze(0)
    if embeddings_a.shape[0] != embeddings_b.shape[0]:
        raise ValueError("CKA requires the same number of samples in both embeddings.")
    if embeddings_a.shape[0] < 2:
        raise ValueError("CKA requires at least two samples (n >= 2).")

    embeddings_a = embeddings_a.float()
    embeddings_b = embeddings_b.float()

    gram_a = embeddings_a @ embeddings_a.T
    gram_b = embeddings_b @ embeddings_b.T
    gram_a = _center_gram(gram_a)
    gram_b = _center_gram(gram_b)

    hsic = (gram_a * gram_b).sum()
    denom = torch.norm(gram_a) * torch.norm(gram_b)
    return hsic / (denom + eps)

class TextEmbedder:
    def __init__(self, model_name="bert", device="cuda"):
        self.model_name = model_name
        self.device = device
        if model_name == "align": 
            # Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
            # https://proceedings.mlr.press/v139/jia21b/jia21b.pdf
            self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
            log.info(f"Initialized Align Model for text embedding.")
        elif model_name == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            log.info(f"Initialized CLIP Text Model for text embedding.")
        elif model_name == "siglip":
            self.model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224") # SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224") # 
            log.info(f"Initialized SigLIP Model for text embedding.")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported options are 'align', 'clip', and 'siglip'.")

    def embed(self, text):
        inputs = self.tokenizer(text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=self.tokenizer.model_max_length
                                ).to(self.device) # 'pt' indicates that hte tokenizer should return PyTorch tensors
        # print(inputs)
        with torch.no_grad():
            if self.model_name == "align":
                outputs = self.model.get_text_features(**inputs)
                return outputs
            elif self.model_name == "clip":
                outputs = self.model(**inputs)
            elif self.model_name == "siglip":
                outputs = self.model(**inputs)
        return outputs.pooler_output  # Return last_hidden_state[:, 0, :]
    
@torch.no_grad()
def encode_classes(classnames, templates, embedder, device="cuda"):
    all_embeddings = []
    
    for name in classnames:
        texts = [t.format(name.replace("_", " ")) for t in templates]
        embeds = embedder.embed(texts) # (T, D)
        embeds = F.normalize(embeds, dim=-1)
        mean_embed = embeds.mean(dim=0)
        mean_embed = F.normalize(mean_embed, dim=-1)
        all_embeddings.append(mean_embed)

    return torch.stack(all_embeddings, dim=0)  # (num_classes, D)

def find_closest_imagenet_classes(cifar_embeds, imagenet_embeds, imagenet_classnames, topk=1, return_indices=False):
    similarity = cifar_embeds @ imagenet_embeds.T  # (Ncifar, Nimagenet)
    values, indices = similarity.topk(topk, dim=-1)

    closest_classes = []
    for i in range(cifar_embeds.shape[0]):
        closest = []
        for rank in range(topk):
            idx = indices[i, rank].item()
            if return_indices:
                closest.append((idx, values[i, rank].item()))
            else:
                closest.append((imagenet_classnames[idx], values[i, rank].item()))
        closest_classes.append(closest)
    
    return closest_classes

if __name__ == "__main__":
    embedder = TextEmbedder(model_name="clip", device="cuda")  # or "clip"

    templates = [
        "a photo of a {}",
        "an image of a {}",
        "a picture of a {}",
        # "a blurry photo of a {}",
        # "a black and white photo of a {}",
        "a cropped photo of a {}",
        "a close-up photo of a {}",
        "a bright photo of a {}",
    ]

    cifar_embeds = encode_classes(cifar10_classes, templates, embedder)
    imagenet_embeds = encode_classes(imagenet_classes, templates, embedder)
    
    similarity = cifar_embeds @ imagenet_embeds.T  # (Ncifar, Nimagenet)
    topk = 5
    values, indices = similarity.topk(topk, dim=-1)

    for i, cifar_name in enumerate(cifar10_classes):
        print(f"\nCIFAR class: {cifar_name}")
        for rank in range(topk):
            idx = indices[i, rank].item()
            print(f"  {rank+1}. {imagenet_classes[idx]} (sim={values[i, rank].item():.4f})")
