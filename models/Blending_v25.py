from models.Blending import Blending


class Blending_v8(Blending):
    """
    v8 rollback wrapper.

    This keeps the baseline Blending behavior and the v8 parser flags, but does
    not add any extra hard-preserve or latent-style-color path.
    """

    pass
