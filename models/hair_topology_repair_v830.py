"""Conservative interior-only topology repair for V2.30 hair masks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, radius)


class HairTopologyRepairV830(nn.Module):
    """Accept small surrounded holes without growing the exterior contour."""

    def __init__(self, hole_radius: int = 2, neighbor_threshold: float = 0.75):
        super().__init__()
        self.hole_radius = int(hole_radius)
        self.neighbor_threshold = float(neighbor_threshold)

    def config_dict(self) -> dict[str, object]:
        return {
            "hole_radius": self.hole_radius,
            "neighbor_threshold": self.neighbor_threshold,
            "interior_only": True,
            "face_hole_default": "REJECT",
        }

    def forward(
        self,
        *,
        target_hair_mask: torch.Tensor,
        source_face_mask: torch.Tensor,
        pp_hair_evidence: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        hair = (target_hair_mask >= 0.5).to(target_hair_mask.dtype)
        closed = _erode(_dilate(hair, self.hole_radius), self.hole_radius)
        candidate = (closed - hair).clamp(0, 1)
        local_fraction = F.avg_pool2d(hair, 5, stride=1, padding=2)

        left = torch.zeros_like(hair)
        right = torch.zeros_like(hair)
        up = torch.zeros_like(hair)
        down = torch.zeros_like(hair)
        for offset in range(1, self.hole_radius + 1):
            left[..., :, offset:] = torch.maximum(
                left[..., :, offset:], hair[..., :, :-offset]
            )
            right[..., :, :-offset] = torch.maximum(
                right[..., :, :-offset], hair[..., :, offset:]
            )
            up[..., offset:, :] = torch.maximum(
                up[..., offset:, :], hair[..., :-offset, :]
            )
            down[..., :-offset, :] = torch.maximum(
                down[..., :-offset, :], hair[..., offset:, :]
            )
        direction_count = left + right + up + down
        surrounded = (direction_count >= 3.0).to(hair.dtype)
        accepted_pre_face = candidate * (local_fraction >= self.neighbor_threshold).to(
            hair.dtype
        ) * surrounded
        face = (source_face_mask >= 0.5).to(hair.dtype)
        if pp_hair_evidence is None:
            face_override = torch.zeros_like(hair)
        else:
            face_override = (pp_hair_evidence >= 0.85).to(hair.dtype)
        rejected_face = accepted_pre_face * face * (1.0 - face_override)
        accepted = accepted_pre_face * (1.0 - face + face_override).clamp(0, 1)
        accepted_face = accepted * face
        repaired = (hair + accepted).clamp(0, 1)
        if not return_aux:
            return repaired
        pixels = float(hair.shape[-2] * hair.shape[-1])
        return repaired, {
            "original_hair_mask": hair,
            "closed_hair_mask": closed,
            "hole_candidate": candidate,
            "accepted_holes": accepted,
            "rejected_face_holes": rejected_face,
            "accepted_face_holes": accepted_face,
            "neighbor_hair_fraction": local_fraction,
            "surrounding_direction_count": direction_count,
            "hole_candidate_fraction": candidate.mean(dim=(1, 2, 3)),
            "accepted_hole_fraction": accepted.mean(dim=(1, 2, 3)),
            "rejected_face_hole_fraction": rejected_face.mean(dim=(1, 2, 3)),
            "accepted_face_hole_fraction": accepted_face.mean(dim=(1, 2, 3)),
            # Conservative upper bound; exact components are audited offline.
            "max_component_size_upper_bound": accepted.flatten(1).sum(dim=1),
            "pixel_count": torch.full(
                (hair.size(0),), pixels, device=hair.device, dtype=hair.dtype
            ),
        }


__all__ = ["HairTopologyRepairV830"]
