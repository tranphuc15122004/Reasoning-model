import torch
import torch.nn as nn
from typing import Optional, List

# Minimal fusion model: vision encoder -> projector -> concat with text embeddings -> pass to LLM via inputs_embeds.
# This is a scaffold to fine-tune later; no training loops included here.

class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or max(vision_dim, llm_dim)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisionPerceiverResampler(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_latents: int = 64,
        depth: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.llm_dim = llm_dim
        self.proj_in = nn.Linear(vision_dim, llm_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, llm_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_q": nn.LayerNorm(llm_dim),
                "ln_kv": nn.LayerNorm(llm_dim),
                "attn": nn.MultiheadAttention(embed_dim=llm_dim, num_heads=heads, dropout=dropout, batch_first=True),
                "ln_ff": nn.LayerNorm(llm_dim),
                "ff": nn.Sequential(
                    nn.Linear(llm_dim, llm_dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(llm_dim * ff_mult, llm_dim),
                    nn.Dropout(dropout),
                ),
            }) for _ in range(depth)
        ])

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        # vision_feats: [B, Sv, Dv]
        x = self.proj_in(vision_feats)  # [B, Sv, Dl]
        B = x.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1).to(x.dtype)  # [B, N, Dl]
        for blk in self.layers:
            q = blk["ln_q"](lat)
            kv = blk["ln_kv"](x)
            attn_out, _ = blk["attn"](q, kv, kv, need_weights=False)
            lat = lat + attn_out
            lat = lat + blk["ff"](blk["ln_ff"](lat))
        return lat  # [B, N, Dl]


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        llm,  # transformers AutoModelForCausalLM or compatible
        llm_tokenizer,
    projector: nn.Module,
        image_token: str = "<image>",
        freeze_vision: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.vision = vision_encoder
        self.llm = llm
        self.tokenizer = llm_tokenizer
        self.projector = projector
        self.image_token = image_token

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # Ensure the image token exists in tokenizer
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
            try:
                self.llm.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                pass

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # For CLIPVisionModel, forward returns BaseModelOutputWithPooling, use last_hidden_state [B, Sv, Dv]
        out = self.vision(images)
        feats = out.last_hidden_state
        return feats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs,
    ):
        # Build text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if images is not None:
            vision_feats = self.encode_image(images)  # [B, Sv, Dv]
            proj_feats = self.projector(vision_feats)  # [B, Sv or N, Dl]

            # Simple strategy: prepend projected vision tokens at the position of the image token.
            # Find <image> token positions; for simplicity assume exactly one per sample.
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            B, T, Dl = inputs_embeds.shape
            fused_embeds: List[torch.Tensor] = []
            for b in range(B):
                ids_b = input_ids[b]
                embeds_b = inputs_embeds[b]
                pos = (ids_b == image_token_id).nonzero()
                if len(pos) == 0:
                    fused = torch.cat([proj_feats[b], embeds_b], dim=0)
                else:
                    p = pos[0].item()
                    fused = torch.cat([embeds_b[:p], proj_feats[b], embeds_b[p + 1 :]], dim=0)
                fused_embeds.append(fused)

            # Pad to same length
            max_len = max(x.shape[0] for x in fused_embeds)
            padded = []
            new_labels = []
            for b, x in enumerate(fused_embeds):
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    x = torch.cat([x, x.new_zeros(pad_len, x.shape[1])], dim=0)
                padded.append(x)
                if labels is not None:
                    lb = labels[b]
                    if lb.shape[0] < max_len:
                        pad_val = -100
                        lb = torch.cat([lb, lb.new_full((max_len - lb.shape[0],), pad_val)], dim=0)
                    new_labels.append(lb)
            inputs_embeds = torch.stack(padded, dim=0)
            if labels is not None:
                labels = torch.stack(new_labels, dim=0)

            # Adjust attention mask
            if attention_mask is not None:
                att = []
                for b in range(len(fused_embeds)):
                    l = fused_embeds[b].shape[0]
                    mask = inputs_embeds.new_ones(l)
                    att.append(mask)
                attention_mask = torch.stack(att, dim=0)

        if max_new_tokens is not None:
            # Generation path via model.generate using inputs_embeds
            return self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        # Training path
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out
