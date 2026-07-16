import torch


ENCODER_TAIL_START_V8 = 6


def clamp_style_tail_start_v8(tail_start: int, latent_layers: int, base_start: int = ENCODER_TAIL_START_V8) -> int:
    tail_start = int(tail_start)
    return max(base_start, min(tail_start, int(latent_layers)))


def blend_style_color_tail_v8(
    latent_face_s: torch.Tensor,
    latent_color_s: torch.Tensor,
    raw_tail_s: torch.Tensor,
    *,
    tail_start: int = 8,
    blend_strength: float = 1.0,
    reference_blend: float = 0.0,
    base_start: int = ENCODER_TAIL_START_V8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compose the v8 hair-color latent.

    The pretrained blending encoder was built for S[6:], so it still receives and
    returns that 12-layer tail. We only write the color residual into S[tail_start:],
    keeping earlier structure-sensitive style layers anchored to the SATD result.
    """
    if latent_face_s.shape != latent_color_s.shape:
        raise ValueError(
            f"latent_face_s and latent_color_s must have the same shape, got "
            f"{tuple(latent_face_s.shape)} vs {tuple(latent_color_s.shape)}"
        )

    latent_layers = latent_face_s.size(1)
    base_start = int(base_start)
    if base_start < 0 or base_start >= latent_layers:
        raise ValueError(f"Invalid base_start={base_start} for latent with {latent_layers} layers")

    base_tail = latent_face_s[:, base_start:]
    color_tail = latent_color_s[:, base_start:]
    if raw_tail_s.shape != base_tail.shape:
        raise ValueError(f"raw_tail_s shape {tuple(raw_tail_s.shape)} does not match S[{base_start}:] {tuple(base_tail.shape)}")

    tail_start = clamp_style_tail_start_v8(tail_start, latent_layers, base_start=base_start)
    rel_start = tail_start - base_start
    composed_tail = base_tail.clone()
    if rel_start >= composed_tail.size(1):
        empty_delta = composed_tail[:, :0]
        return latent_face_s, empty_delta

    blend_strength = float(blend_strength)
    reference_blend = float(reference_blend)
    candidate = base_tail[:, rel_start:] + blend_strength * (raw_tail_s[:, rel_start:] - base_tail[:, rel_start:])
    if reference_blend > 0:
        candidate = torch.lerp(candidate, color_tail[:, rel_start:], min(reference_blend, 1.0))

    composed_tail[:, rel_start:] = candidate
    composed_s = torch.cat((latent_face_s[:, :base_start], composed_tail), dim=1)
    style_delta = candidate - base_tail[:, rel_start:]
    return composed_s, style_delta
