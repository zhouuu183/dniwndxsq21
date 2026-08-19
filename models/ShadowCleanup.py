import torch
import torch.nn.functional as F
from torch import nn


class ShadowCleanup(nn.Module):
    """
    Multi-branch ROI-gated shadow cleanup:
    - ring branch: harmonizes low-frequency shadow around the transferred hair boundary
    - halo branch: removes leftover dark hair halos near ears / neck / forehead
    - tail branch: suppresses sparse residual strands under short-hair targets
    - source-copy branch: selectively copies reliable source pixels back in visible non-hair areas
    - body-protect branch: preserves shoulders / clothes / neck structure from accidental over-cleanup
    """

    def __init__(self, opts=None):
        super().__init__()
        self.default_strength = 0.75 if opts is None else getattr(opts, "shadow_cleanup_strength", 0.75)
        self.default_source_blend = 0.55 if opts is None else getattr(opts, "shadow_cleanup_source_blend", 0.55)
        self.default_kernel = 21 if opts is None else getattr(opts, "shadow_cleanup_kernel", 21)
        self.default_ring_strength = 0.65 if opts is None else getattr(opts, "shadow_cleanup_ring_strength", 0.65)
        self.default_halo_strength = 0.95 if opts is None else getattr(opts, "shadow_cleanup_halo_strength", 0.95)
        self.default_tail_strength = 0.55 if opts is None else getattr(opts, "shadow_cleanup_tail_strength", 0.55)
        self.default_copy_strength = 0.70 if opts is None else getattr(opts, "shadow_cleanup_copy_strength", 0.70)
        self.default_body_protect_strength = 0.45 if opts is None else getattr(opts, "shadow_cleanup_body_protect_strength", 0.45)

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeezed = image.dim() == 3
        if squeezed:
            image = image.unsqueeze(0)
        return image, squeezed

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(mask.float(), size=size, mode="nearest")

    @staticmethod
    def _masked_box_blur(image: torch.Tensor, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        if mask.shape[1] != 1:
            mask = mask[:, :1]

        numer = F.avg_pool2d(image * mask, kernel_size=kernel_size, stride=1, padding=padding)
        denom = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding).clamp(min=1e-4)
        return numer / denom

    @torch.inference_mode()
    def forward(
        self,
        source_image: torch.Tensor,
        base_image: torch.Tensor,
        cleanup_masks: dict[str, torch.Tensor],
        strength: float | None = None,
        source_blend: float | None = None,
        kernel_size: int | None = None,
        ring_strength: float | None = None,
        halo_strength: float | None = None,
        tail_strength: float | None = None,
        copy_strength: float | None = None,
        body_protect_strength: float | None = None,
    ) -> torch.Tensor:
        source_image, source_squeezed = self._ensure_batch(source_image)
        base_image, base_squeezed = self._ensure_batch(base_image)

        if source_image.shape[-2:] != base_image.shape[-2:]:
            source_image = F.interpolate(source_image, size=base_image.shape[-2:], mode="bilinear", align_corners=False)

        size = base_image.shape[-2:]
        shadow_ring = self._resize_mask(cleanup_masks["M_shadow_ring"], size)
        remove_halo = self._resize_mask(cleanup_masks["M_remove_halo"], size)
        remove_tail = self._resize_mask(cleanup_masks["M_remove_tail"], size)
        source_visible = self._resize_mask(cleanup_masks["M_source_visible"], size)
        source_copy = self._resize_mask(cleanup_masks["M_source_copy"], size)
        cleanup = self._resize_mask(cleanup_masks["M_cleanup"], size)
        body_preserve = self._resize_mask(cleanup_masks["M_body_preserve"], size)

        strength = self.default_strength if strength is None else strength
        source_blend = self.default_source_blend if source_blend is None else source_blend
        kernel_size = self.default_kernel if kernel_size is None else kernel_size
        ring_strength = self.default_ring_strength if ring_strength is None else ring_strength
        halo_strength = self.default_halo_strength if halo_strength is None else halo_strength
        tail_strength = self.default_tail_strength if tail_strength is None else tail_strength
        copy_strength = self.default_copy_strength if copy_strength is None else copy_strength
        body_protect_strength = self.default_body_protect_strength if body_protect_strength is None else body_protect_strength

        lowfreq_src = self._masked_box_blur(source_image, source_visible, kernel_size)
        lowfreq_base = self._masked_box_blur(base_image, source_visible, kernel_size)
        color_delta = lowfreq_src - lowfreq_base

        cleanup_core = (
            ring_strength * shadow_ring
            + halo_strength * remove_halo
            + tail_strength * remove_tail
        ).clamp(0, 1)
        cleanup_alpha = (strength * cleanup * cleanup_core * (1 - 0.80 * body_preserve)).clamp(0, 1)
        corrected = (base_image + cleanup_alpha * color_delta).clamp(0, 1)

        copy_core = (
            0.45 * shadow_ring
            + 0.90 * remove_halo
            + 0.55 * remove_tail
            + 0.65 * source_copy
        ).clamp(0, 1)
        copy_alpha = (source_blend * copy_strength * copy_core * source_visible * (1 - 0.88 * body_preserve)).clamp(0, 1)
        corrected = corrected * (1 - copy_alpha) + source_image * copy_alpha

        body_alpha = (body_protect_strength * source_blend * body_preserve * source_visible).clamp(0, 1)
        corrected = corrected * (1 - body_alpha) + source_image * body_alpha
        corrected = corrected.clamp(0, 1)

        if source_squeezed and base_squeezed:
            return corrected[0]
        return corrected
