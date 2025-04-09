from finetrainers.models.flux import FluxModelSpecification


class DummyFluxModelSpecification(FluxModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-flux-pipe", **kwargs)
