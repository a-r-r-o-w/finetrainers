def patch_diffusers_rms_norm_forward() -> None:
    import diffusers.models.normalization

    from .rms_norm import _patched_rms_norm_forward

    diffusers.models.normalization.RMSNorm.forward = _patched_rms_norm_forward
