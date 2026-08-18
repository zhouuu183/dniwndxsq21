from __future__ import annotations

import torch

from models.Embedding import Embedding as EmbeddingBaseline


class EmbeddingV5(EmbeddingBaseline):
    """V5 embedding wrapper for native-resolution accessory recovery.

    The shared ``Embedding`` implementation remains the baseline contract.
    V5 keeps its source-face metadata under a V5-specific key so the native
    earring compositor cannot change the inputs used by the other pipelines.
    """

    @torch.inference_mode()
    def embedding_images(self, images_to_name: dict[torch.Tensor, list[str]], **kwargs):
        name_to_embed = super().embedding_images(images_to_name, **kwargs)

        face_image = None
        for image, names in images_to_name.items():
            if "face" in names:
                face_image = image
                break
        if face_image is None:
            raise KeyError("EmbeddingV5 requires a face image named 'face'.")

        if face_image.ndim == 3:
            face_image = face_image.unsqueeze(0)
        if face_image.ndim != 4 or face_image.size(1) != 3:
            raise ValueError(
                "EmbeddingV5 face input must have shape [C,H,W] or [B,3,H,W]."
            )
        if face_image.dtype == torch.uint8:
            face_image = face_image.float().div(255.0)
        face_image = face_image.to(device=self.opts.device)

        face_record = name_to_embed["face"]
        face_record["image_v5_native"] = face_image
        return name_to_embed
