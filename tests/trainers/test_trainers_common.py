from finetrainers import Trainer
from huggingface_hub import snapshot_download
import tempfile
import os

class TrainerTestMixin:
    def get_training_args(self):
        raise NotImplementedError
    
    def download_dataset_txt_format(self, cache_dir):
        path = snapshot_download(repo_id="finetrainers/dummy-disney-dataset", repo_type="dataset", cache_dir=cache_dir)
        return path
    
    def test_precomputation_txt_format(self):
        # Here we assume the dataset is formatted like:
        # https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset/tree/main
        training_args = self.get_training_args()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args.data_root = self.download_dataset_txt_format(cache_dir=tmpdir)
            trainer = Trainer(training_args)
            training_args = trainer.args

            trainer.prepare_dataset()
            trainer.prepare_models()
            trainer.prepare_precomputations()

            precomputed_dir = os.path.join(training_args.data_root, f"{training_args.pretrained_model_name_or_path}_precomputed")
            assert os.path.exists(precomputed_dir), f"Precomputation wasn't successful. Couldn't find the precomputed dir: {os.listdir()}"