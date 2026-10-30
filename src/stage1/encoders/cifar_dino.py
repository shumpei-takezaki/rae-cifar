import torch
from torch import nn

from cifar_dino.export import load_backbone

from . import register_encoder


@register_encoder()
class CifarDinoViT(nn.Module):
    """Frozen dino-cifar teacher backbone (ViT-Tiny/4, 32x32 -> 8x8 patch tokens of dim 192).

    ``normalize=True`` drops the final LayerNorm's affine, as ``Dinov2withNorm`` does for the official RAE.
    Returns patch tokens only (CLS and register tokens are dropped).
    """

    image_mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 stats used by dino-cifar
    image_std = (0.2470, 0.2435, 0.2616)

    def __init__(self, ckpt_path: str, normalize: bool = True):
        super().__init__()
        self.encoder, self.meta = load_backbone(ckpt_path)
        if normalize:
            self.encoder.norm = nn.LayerNorm(self.encoder.embed_dim, eps=self.encoder.norm.eps, elementwise_affine=False)
        self.encoder.requires_grad_(False)
        self.patch_size = self.encoder.patch_size
        self.hidden_size = self.encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)["patch_tokens"]
