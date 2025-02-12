# Run: python3 tests/test_dataset.py

import sys

import finetrainers

def test_video_dataset():
    from finetrainers.dataset import ImageOrVideoDataset

    dataset_dirs = ImageOrVideoDataset(
        data_root="assets/tests/",
        caption_column="prompts.txt",
        video_column="videos.txt",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )
    dataset_csv = ImageOrVideoDataset(
        data_root="assets/tests/",
        dataset_file="assets/tests/metadata.csv",
        caption_column="caption",
        video_column="video",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )

    assert len(dataset_dirs) == 1
    assert len(dataset_csv) == 1
    assert dataset_dirs[0]["video"].shape == (49, 3, 480, 720)
    assert (dataset_dirs[0]["video"] == dataset_csv[0]["video"]).all()

    print(dataset_dirs[0]["video"].shape)


def test_video_dataset_with_resizing():
    from finetrainers.dataset import ImageOrVideoDatasetWithResizing

    dataset_dirs = ImageOrVideoDatasetWithResizing(
        data_root="assets/tests/",
        caption_column="prompts.txt",
        video_column="videos.txt",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )
    dataset_csv = ImageOrVideoDatasetWithResizing(
        data_root="assets/tests/",
        dataset_file="assets/tests/metadata.csv",
        caption_column="caption",
        video_column="video",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )

    assert len(dataset_dirs) == 1
    assert len(dataset_csv) == 1
    assert dataset_dirs[0]["video"].shape == (49, 3, 480, 720)  # Changes due to T2V frame bucket sampling
    assert (dataset_dirs[0]["video"] == dataset_csv[0]["video"]).all()

    print(dataset_dirs[0]["video"].shape)


def test_video_dataset_with_bucket_sampler():
    import torch
    from finetrainers.dataset import BucketSampler, ImageOrVideoDatasetWithResizing
    from torch.utils.data import DataLoader

    dataset_dirs = ImageOrVideoDatasetWithResizing(
        data_root="assets/tests/",
        caption_column="prompts_multi.txt",
        video_column="videos_multi.txt",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )
    sampler = BucketSampler(dataset_dirs, batch_size=8)

    def collate_fn(data):
        captions = [x["prompt"] for x in data[0]]
        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos)
        return captions, videos

    dataloader = DataLoader(dataset_dirs, batch_size=1, sampler=sampler, collate_fn=collate_fn)
    first = False

    for captions, videos in dataloader:
        if not first:
            assert len(captions) == 2 and isinstance(captions[0], str)
            assert videos.shape == (2, 49, 3, 480, 720)
            first = True
        else:
            assert len(captions) == 2 and isinstance(captions[0], str)
            assert videos.shape == (2, 49, 3, 256, 360)
            break

def test_load_json_dataset():
    from finetrainers.dataset import ImageOrVideoDataset
    
    dataset_json = ImageOrVideoDataset(
        data_root="assets/tests/",
        dataset_file="assets/tests/dummy_json_dataset.json",
        caption_column="caption",
        video_column="path",
        resolution_buckets=[(49, 480, 720)],
        id_token=None,
    )

    assert len(dataset_json.video_paths) == 2
    assert len(dataset_json.prompts) == 2
    assert str(dataset_json.video_paths[0]) == "assets/tests/videos/hiker.mp4"
    assert dataset_json.prompts[0] == "This is a test caption"
    assert str(dataset_json.video_paths[1]) == "assets/tests/videos/hiker_tiny.mp4"
    assert dataset_json.prompts[1] == "This is another test caption"
  
if __name__ == "__main__":
    sys.path.append("./training")

    test_video_dataset()
    test_video_dataset_with_resizing()
    test_video_dataset_with_bucket_sampler()
