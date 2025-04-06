import pathlib
import sys


project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers.models.flux import FluxModelSpecification  # noqa


class DummyFluxModelSpecification(FluxModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(pretrained_model_name_or_path="hf-internal-testing/tiny-flux-pipe", **kwargs)
