from finetrainers.models.hidream import HiDreamImageModelSpecification


class DummyHiDreamImageModelSpecification(HiDreamImageModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(
            pretrained_model_name_or_path="./dump_hidream_pipeline",
            # This is passed separately so that AutoTokenizer can load T5TokenizerFast instead of the BaseSpec's
            # T5Tokenier from the internal testing repo.
            tokenizer_3_id="hf-internal-testing/tiny-random-t5",
            **kwargs,
        )
