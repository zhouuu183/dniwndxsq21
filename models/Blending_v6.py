from __future__ import annotations

import argparse

import torch
from torch import nn

from models.Blending import Blending as AuthorBlending
from models.Net import Net
from models.postprocess_v6 import PostProcessModelV6, load_checkpoint_compat
from utils.save_utils import save_gen_image, save_latents


class BlendingV6(nn.Module):
    """Author pre-PP transfer with a live PP decode and isolated SATD branch.

    The base path keeps the author hair-shape alignment and colour blending,
    then runs SATD on the blended ``F`` feature and decodes
    ``I_satd_blend``.  Its 256px image is the direct PP input for one StyleGAN
    decode.  SATD is therefore not applied again after PP.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(opts) if net is None else net

        # Reuse the original blending encoder, mask operator and downsampler.
        # Its original PP object is intentionally not called: V6 must decode
        # its own loaded PP model at runtime so training and inference have
        # the same source + I_blend_256 -> S/F -> StyleGAN path.
        author_checkpoint_path = getattr(opts, "pp_checkpoint", None)
        if not author_checkpoint_path:
            raise ValueError("pp_checkpoint is required for the author PP core.")
        self.author_blending = AuthorBlending(opts, net=self.net).eval()

        # ``_author_transfer`` produces the exact author pre-PP target used
        # by the live V6 PP call below.
        self.blending_encoder = self.author_blending.blending_encoder
        self.dilate_erosion = self.author_blending.dilate_erosion
        self.downsample_256 = self.author_blending.downsample_256

        # The V6 model owns the live, author-compatible PP decode. Its final
        # compositor only writes the verified earring alpha after that decode;
        # direct-SATD mode deliberately has no second background residual.
        pp_args = argparse.Namespace(**vars(opts))
        pp_args.use_mod = getattr(
            opts,
            "pp_v6_use_mod",
            getattr(opts, "pp_v5_use_mod", True),
        )
        pp_args.use_full = getattr(
            opts,
            "pp_v6_use_full",
            getattr(opts, "pp_v5_use_full", True),
        )
        pp_args.pretrain = False
        pp_args.direct_satd_pp_input = bool(
            getattr(opts, "direct_satd_pp_input", True)
        )
        pp_args.finetune = False
        pp_args.enable_direct_face_skin_restore = False
        pp_args.enable_native_source_detail = False
        self.post_process = PostProcessModelV6(pp_args).to(opts.device).eval()

        pp_checkpoint = (
            getattr(opts, "pp_v6_checkpoint", None)
            or getattr(opts, "pp_v5_checkpoint", None)
            or author_checkpoint_path
        )
        checkpoint = load_checkpoint_compat(pp_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        result = self.post_process.load_state_dict(state_dict, strict=False)
        missing_base = [
            key for key in result.missing_keys
            if not key.startswith(
                (
                    "hf_extractor",
                    "mask_refresher",
                    "brightness_reestimator",
                    "ear_injector_64",
                    "ear_injector_128",
                    "native_source_detail_conditioner",
                    "native_source_detail_injector_128",
                )
            )
        ]
        if missing_base:
            print(f"[BlendingV6] Missing PP base keys: {len(missing_base)}")
            print(missing_base[:20])

    @torch.inference_mode()
    def _author_transfer(self, align_shape, align_color, name_to_embed):
        """Reproduce the no-suffix ``Blending.blend_images`` pre-PP state."""
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat(
                [name_to_embed[name]["mask"] for name in ("face", "color")],
                dim=0,
            )
        )
        HM_1D = mask_de[0][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)
        HM_X = align_color["HM_X"]
        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        if I_1 is not I_3 or I_1 is not I_2:
            S_blend_6_18 = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * HM_3E,
            )
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        # Keep a replay point for the post-author SATD candidate.  Both
        # renders must use identical StyleGAN noise; otherwise their RGB
        # difference contains unrelated stochastic hair/background texture
        # and the SATD residual becomes both weak and visibly streaked.
        self._v6_author_rng_cpu = torch.random.get_rng_state()
        self._v6_author_rng_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_shape["latent_F_align"],
        )
        self._v6_author_rng_after_cpu = torch.random.get_rng_state()
        self._v6_author_rng_after_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        return I_1, HM_3E, HM_X, target_mask, S_blend, I_blend

    @torch.inference_mode()
    def _render_satd_candidate(self, S_blend, I_blend, kwargs):
        """Render SATD only after the author image path has consumed its noise.

        StyleGAN defaults to sampled per-layer noise.  Constructing a SATD
        candidate before the author transfer/PP decode changes that RNG
        sequence even when SATD never enters S/F.  Resolve the side branch
        lazily so the author image remains bit-for-bit on its own execution
        schedule.
        """
        satd_alignment = kwargs.get("satd_alignment")
        if satd_alignment is None:
            factory = kwargs.get("satd_alignment_factory")
            if callable(factory):
                # SATD feature preparation may run SEAN/StyleGAN helpers and
                # consume random noise.  It is an auxiliary branch; restore
                # the author's post-transfer RNG state immediately so the
                # subsequent PP decode keeps the same noise schedule.
                factory_cpu = torch.random.get_rng_state()
                factory_cuda = (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                )
                satd_alignment = factory()
                torch.random.set_rng_state(factory_cpu)
                if factory_cuda is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(factory_cuda)

        if (
            isinstance(satd_alignment, dict)
            and bool(satd_alignment.get("satd_applied", False))
            and torch.is_tensor(satd_alignment.get("latent_F_satd"))
        ):
            # Replay the author's noise for this side render, then restore the
            # post-author RNG state so SATD does not perturb the later PP
            # decode.  This changes only the SATD comparison, not the author
            # image or the PP noise schedule.
            current_cpu = torch.random.get_rng_state()
            current_cuda = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                before_cpu = getattr(self, "_v6_author_rng_cpu", None)
                before_cuda = getattr(self, "_v6_author_rng_cuda", None)
                if before_cpu is not None:
                    torch.random.set_rng_state(before_cpu)
                if before_cuda is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(before_cuda)
                candidate, _ = self.net.generator(
                    [S_blend],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=satd_alignment["latent_F_satd"],
                )
            finally:
                torch.random.set_rng_state(current_cpu)
                if current_cuda is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(current_cuda)
            cleanup_masks = {
                key: value
                for key, value in satd_alignment.get("delta_masks", {}).items()
                if key in {
                    "M_boundary",
                    "M_remove",
                    "M_remove_halo",
                    "M_remove_tail",
                    "M_remove_face",
                    "M_remove_neck",
                    "M_remove_context",
                }
                and torch.is_tensor(value)
            }
            return candidate, cleanup_masks

        return I_blend, {}

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1, HM_3E, HM_X, target_mask, S_blend, I_blend = self._author_transfer(
            align_shape,
            align_color,
            name_to_embed,
        )
        # SATD is a separate F-space correction on top of the author's
        # already-completed shape/colour latent.  The corrected image is the
        # actual PP input in direct-SATD mode; the baseline I_blend is kept
        # only as a diagnostic/reference image.
        I_blend_satd, cleanup_masks = self._render_satd_candidate(
            S_blend,
            I_blend,
            kwargs,
        )
        direct_satd_pp_input = bool(
            getattr(self.opts, "direct_satd_pp_input", True)
        )
        pp_input_256 = (
            self.downsample_256(I_blend_satd)
            if direct_satd_pp_input
            else self.downsample_256(I_blend)
        )

        return_stage = kwargs.get("return_stage")
        stop_before_pp = bool(kwargs.get("stop_before_pp", False))
        if return_stage is not None and return_stage not in {"color_before_pp", "before_pp"}:
            raise ValueError(
                "BlendingV6 return_stage must be 'color_before_pp'/'before_pp', "
                f"got {return_stage!r}."
            )
        if stop_before_pp or return_stage is not None:
            # Direct-SATD dataset contract: the PP input itself is the SATD
            # image.  Keep the same tensor in the legacy high-resolution
            # fields so no later stage can accidentally mix a different target
            # with the direct-SATD input.
            base_01 = ((pp_input_256[0] + 1.0) * 0.5).clamp(0, 1).detach().cpu()
            return {
                "stage": "color_before_pp",
                "image": base_01,
                "color_before_pp": base_01,
                "pre_reference_color": base_01,
                "completed_hair_highres": ((I_blend_satd[0] + 1.0) * 0.5).clamp(0, 1).detach().cpu(),
                "satd_background_highres": ((I_blend_satd[0] + 1.0) * 0.5).clamp(0, 1).detach().cpu(),
                "cleanup_masks": cleanup_masks,
                "target_hair_mask": HM_X[0].detach().cpu(),
                "direct_satd_pp_input": direct_satd_pp_input,
            }

        # This is the only PP route: source plus the author pre-PP transfer
        # become one S/F pair, then one StyleGAN decode.  Neither SATD nor
        # earring RGB is present in that encoder/decode path.
        latent_s, latent_f, aux = self.post_process(
            I_1,
            pp_input_256,
            target_mask,
            HM_3E,
            target_hair_mask=HM_X,
            authoritative_hair_highres=I_blend_satd if direct_satd_pp_input else I_blend,
            authoritative_target_highres=I_blend_satd if direct_satd_pp_input else I_blend,
            direct_satd_pp_input=direct_satd_pp_input,
            earring_reference=name_to_embed["face"].get("image_1024"),
            source_face_reference=name_to_embed["face"].get("image_1024"),
        )
        pp_image, _ = self.net.generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=latent_f,
        )
        # SATD is already present in the PP input.  Do not attach it as a
        # second post-decode residual; that would apply the correction twice.
        aux["direct_satd_pp_input"] = torch.full(
            (I_1.size(0), 1, 1, 1),
            float(direct_satd_pp_input),
            device=I_1.device,
            dtype=I_1.dtype,
        )
        for key, value in cleanup_masks.items():
            aux[key] = value
        I_final, _ = self.post_process.compose_post_decode(pp_image, aux)

        if bool(getattr(self.opts, "save_all", False)):
            output_dir = self.opts.save_all_dir / (kwargs.get("exp_name") or "")
            save_gen_image(output_dir, "BlendingV6", "author_transfer.png", I_blend)
            save_gen_image(output_dir, "BlendingV6", "satd_background_candidate.png", I_blend_satd)
            save_gen_image(output_dir, "FinalV6", "final.png", I_final)
            save_latents(output_dir, "BlendingV6", "blending.npz", S_blend=S_blend)

        return ((I_final[0] + 1.0) * 0.5).clamp(0, 1)
