import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Optional, Literal

try:
    from timm.models import create_model
except Exception as e:
    raise ImportError("timm is required for CAS-ViT encoder. Install with `pip install timm einops`. ") from e


class CASViTFeatureEncoder(nn.Module):
    """
    Wrapper around CAS-ViT (RCViT) to produce token sequences.

    - Uses timm.create_model('<rcvit_variant>', fork_feat=True)
    - Returns SimpleNamespace(last_hidden_state=[B, N, C])
    - Tokenization modes:
        - tokens='avg': global average pooled single token (N=1)
        - tokens='spatial': flatten HxW spatial grid to tokens (N=H*W)
    """

    def __init__(
        self,
        model_name: str = "rcvit_xs",
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        out_stage: int = -1,
        tokens: Literal["avg", "spatial"] = "spatial",
        input_size: int = 224,
    ):
        super().__init__()
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            fork_feat=True,
            input_res=input_size,
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.out_stage = out_stage
        self.tokens = tokens

    @staticmethod
    def _unwrap_state_dict(sd: dict) -> dict:
        # Try common keys
        for k in ("state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
        return sd

    def _load_checkpoint(self, path: str):
        sd = torch.load(path, map_location="cpu")
        sd = self._unwrap_state_dict(sd)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[CASViTFeatureEncoder] Loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        feats = self.model(images)  # list of feature maps [B, C_i, H_i, W_i]
        if not isinstance(feats, (list, tuple)):
            # Safety: some configs could differ; enforce list
            feats = [feats]
        idx = self.out_stage if self.out_stage != -1 else (len(feats) - 1)
        x = feats[idx]
        B, C, H, W = x.shape
        if self.tokens == "avg":
            x = x.mean(dim=(2, 3))  # [B, C]
            x = x.unsqueeze(1)      # [B, 1, C]
        else:
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return SimpleNamespace(last_hidden_state=x)


def build_casvit_encoder(
    model_name: str = "rcvit_xs",
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    tokens: Literal["avg", "spatial"] = "spatial",
    out_stage: int = -1,
    input_size: int = 224,
) -> CASViTFeatureEncoder:
    return CASViTFeatureEncoder(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        out_stage=out_stage,
        tokens=tokens,
        input_size=input_size,
    )
