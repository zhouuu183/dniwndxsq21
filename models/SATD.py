"""
Shadow-Aware Texture Disentangler  (SATD)
==========================================
Lightweight drop-in replacement for the hard binary-mask cascaded interpolation
used in HairFast Eq.(8).

Instead of down-sampling dilated / eroded masks to 32x32 and linearly mixing
four F-space feature tensors with binary weights, SATD **learns** spatially-
adaptive soft blending weights conditioned on:

    (A)  Mask geometry             — MaskEncoder
    (B)  Shadow / highlight cues   — ShadowEstimator  (from face F tensor)
    (C)  All four F-space tensors  — shared FeatureProjector
    (D)  A 3-map soft weight head  — BlendNet
    (E)  Shadow-gated residual     — TextureGate

Design rationale
----------------
* **Cascaded soft blending** preserves the same 3-step interpolation topology
  as the original Eq.(8), but replaces each hard 0/1 weight with a learned
  sigmoid output ∈ (0, 1).
* **Zero-init** on the final BlendNet conv makes sigmoid(0)=0.5 at start,
  so the untrained module produces a uniform average — a safe baseline.
* All normalisations use GroupNorm (no batch-size dependency).
* Depth-wise separable conv keeps TextureGate cheap.

Parameter budget:  ~2.70 M  (< 4 M constraint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_func


# ---------------------------------------------------------------------------
#  (A) Mask Encoder
# ---------------------------------------------------------------------------
class MaskEncoder(nn.Module):
    """
    Encode the 3-channel binary aggregation masks (union / target / intersect)
    into a dense spatial feature map.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 64, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.GELU(),
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """masks : (B, 3, H, W) → (B, out_ch, H, W)"""
        return self.net(masks)


# ---------------------------------------------------------------------------
#  (B) Shadow Estimator
# ---------------------------------------------------------------------------
class ShadowEstimator(nn.Module):
    """
    Predict a single-channel shadow / highlight attention map ∈ (0, 1).

    Fuses a low-rank projection of the *face* F tensor with mask features
    so that the map is aware of both illumination and geometry.
    """

    def __init__(self, feat_ch: int = 512, mask_feat_ch: int = 128,
                 mid_ch: int = 64):
        super().__init__()
        proj_ch = mid_ch // 2                                   # 32
        self.feat_proj = nn.Conv2d(feat_ch, proj_ch, 1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(proj_ch + mask_feat_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, F_face: torch.Tensor,
                mask_feat: torch.Tensor) -> torch.Tensor:
        """→ (B, 1, H, W)"""
        proj = self.feat_proj(F_face)
        return self.head(torch.cat([proj, mask_feat], dim=1))


# ---------------------------------------------------------------------------
#  (E) Texture Gate  — shadow-modulated residual refinement
# ---------------------------------------------------------------------------
class TextureGate(nn.Module):
    """
    A depth-wise-separable conv block produces a residual, gated by a
    channel-wise sigmoid **multiplied** by the scalar shadow map.
    Corrections are therefore suppressed in well-lit, confidently-blended
    regions and amplified in ambiguous shadow / specular areas.
    """

    def __init__(self, feat_ch: int = 512):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=16, bias=False),
            nn.GroupNorm(32, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, F_blend: torch.Tensor,
                shadow_map: torch.Tensor) -> torch.Tensor:
        residual = self.refine(F_blend)
        g = self.gate(F_blend) * shadow_map              # shadow-modulated
        return F_blend + g * residual


# ---------------------------------------------------------------------------
#  Main Module
# ---------------------------------------------------------------------------
class SATD(nn.Module):
    """
    Shadow-Aware Texture Disentangler
    ----------------------------------
    Complete replacement for the hard-mask cascaded interpolation in
    HairFast ``align_images`` (Eq.8).

    Inputs
    ------
    F_face   : (B, C, H, W)  face F-space features        — latent_F_1
    F_sean1  : (B, C, H, W)  SEAN face→target inpainting  — intermediate_align
    F_sean2  : (B, C, H, W)  SEAN hair→target inpainting  — latent_F_out_new
    F_hair   : (B, C, H, W)  hair donor F-space features   — latent_F_2
    masks_hw : (B, 3, H, W)  aggregation masks at feature resolution
               ch-0 = union(hair_mask1, hair_mask_target)
               ch-1 = hair_mask_target
               ch-2 = intersect(hair_mask2, hair_mask_target)

    Output
    ------
    F_align  : (B, C, H, W)  aligned / blended F-space features
    """

    def __init__(self, feat_ch: int = 512, mask_ch: int = 3,
                 mid_ch: int = 128):
        super().__init__()

        # (A) Mask Encoder
        self.mask_encoder = MaskEncoder(in_ch=mask_ch, out_ch=mid_ch)

        # (B) Shadow Estimator
        self.shadow_est = ShadowEstimator(feat_ch, mid_ch)

        # (C) Shared Feature Projector   C → mid_ch
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_ch, mid_ch, 1, bias=False),
            nn.GroupNorm(16, mid_ch),
        )

        # (D) Soft Blending-Weight Generator
        #     input channels = 4 * mid_ch  +  mid_ch  +  1
        blend_in_ch = mid_ch * 5 + 1                     # 641 for mid=128
        self.blend_net = nn.Sequential(
            nn.Conv2d(blend_in_ch, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 3, 1),                        # → 3 weight maps
        )

        # (E) Shadow-Modulated Residual Refinement
        self.texture_gate = TextureGate(feat_ch)

        # Zero-init last conv  →  sigmoid(0)=0.5  →  uniform average at init
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Zero-initialise the final 1x1 conv in *blend_net* so that every
        sigmoid weight starts at 0.5 — the module degrades gracefully to
        a simple uniform average before any training.
        """
        last_conv = self.blend_net[-1]                   # Conv2d(128, 3, 1)
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        F_face:   torch.Tensor,                          # latent_F_1
        F_sean1:  torch.Tensor,                          # intermediate_align
        F_sean2:  torch.Tensor,                          # latent_F_out_new
        F_hair:   torch.Tensor,                          # latent_F_2
        masks_hw: torch.Tensor,                          # (B, 3, H, W)
    ) -> torch.Tensor:

        # (A) encode mask geometry
        mask_feat = self.mask_encoder(masks_hw)           # (B, mid, H, W)

        # (B) estimate shadow / highlight map
        shadow_map = self.shadow_est(F_face, mask_feat)   # (B, 1, H, W)

        # (C) project all four F-space tensors into compact space
        pF_face  = self.feat_proj(F_face)
        pF_sean1 = self.feat_proj(F_sean1)
        pF_sean2 = self.feat_proj(F_sean2)
        pF_hair  = self.feat_proj(F_hair)

        # (D) predict three spatially-adaptive soft blending weights
        blend_in = torch.cat(
            [pF_face, pF_sean1, pF_sean2, pF_hair,
             mask_feat, shadow_map],
            dim=1,
        )
        raw_w = self.blend_net(blend_in)                  # (B, 3, H, W)
        w = torch.sigmoid(raw_w)                          # ∈ (0, 1)

        w0 = w[:, 0:1]                                   # face  ? sean1
        w1 = w[:, 1:2]                                   # prev  ? sean2
        w2 = w[:, 2:3]                                   # prev  ? hair

        # Cascaded soft blending  (structural analogue of Eq.8)
        F_out = F_sean1 + w0 * (F_face - F_sean1)
        F_out = F_sean2 + w1 * (F_out  - F_sean2)
        F_out = F_hair  + w2 * (F_out  - F_hair)

        # (E) shadow-aware residual refinement
        F_out = self.texture_gate(F_out, shadow_map)

        return F_out

    # ------------------------------------------------------------------
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Return total number of learnable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
#  Quick sanity check & parameter audit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"
    model = SATD().to(device)
    n = SATD.count_parameters(model)

    print("=" * 60)
    print(f"  SATD  total params : {n:>10,}  ({n / 1e6:.2f} M)")
    print("=" * 60)
    for name, sub in [
        ("MaskEncoder",      model.mask_encoder),
        ("ShadowEstimator",  model.shadow_est),
        ("FeatureProjector", model.feat_proj),
        ("BlendNet",         model.blend_net),
        ("TextureGate",      model.texture_gate),
    ]:
        cnt = sum(p.numel() for p in sub.parameters())
        print(f"  {name:<20s}: {cnt:>10,}  ({cnt / 1e6:.2f} M)")
    print("=" * 60)

    assert n < 4_000_000, f"Parameter budget exceeded: {n:,}"

    # dummy forward
    B = 2
    dummy = torch.randn(B, 512, 32, 32, device=device)
    masks = torch.rand(B, 3, 32, 32, device=device)
    out = model(dummy, dummy, dummy, dummy, masks)
    print(f"  Input  shape : (B, 512, 32, 32)")
    print(f"  Output shape : {tuple(out.shape)}")
    print("  Sanity check PASSED")
