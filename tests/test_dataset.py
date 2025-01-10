# Run: python3 tests/test_dataset.py

import sys


def test_video_dataset():
    from cogvideox.dataset import VideoDataset

    dataset_dirs = VideoDataset(
        data_root="assets/tests/",
        caption_column="prompts.txt",
        video_column="videos.txt",
        max_num_frames=49,
        id_token=None,
        random_flip=None,
    )
    dataset_csv = VideoDataset(
        data_root="assets/tests/",
        dataset_file="assets/tests/metadata.csv",
        caption_column="caption",
        video_column="video",
        max_num_frames=49,
        id_token=None,
        random_flip=None,
    )

    assert len(dataset_dirs) == 1
    assert len(dataset_csv) == 1
    assert dataset_dirs[0]["video"].shape == (49, 3, 480, 720)
    assert (dataset_dirs[0]["video"] == dataset_csv[0]["video"]).all()

    print(dataset_dirs[0]["video"].shape)


def test_video_dataset_with_resizing():
    from cogvideox.dataset import VideoDatasetWithResizing

    dataset_dirs = VideoDatasetWithResizing(
        data_root="assets/tests/",
        caption_column="prompts.txt",
        video_column="videos.txt",
        max_num_frames=49,
        id_token=None,
        random_flip=None,
    )
    dataset_csv = VideoDatasetWithResizing(
        data_root="assets/tests/",
        dataset_file="assets/tests/metadata.csv",
        caption_column="caption",
        video_column="video",
        max_num_frames=49,
        id_token=None,
        random_flip=None,
    )

    assert len(dataset_dirs) == 1
    assert len(dataset_csv) == 1
    assert dataset_dirs[0]["video"].shape == (48, 3, 480, 720)  # Changes due to T2V frame bucket sampling
    assert (dataset_dirs[0]["video"] == dataset_csv[0]["video"]).all()

    print(dataset_dirs[0]["video"].shape)


def test_video_dataset_with_bucket_sampler():
    import torch
    from cogvideox.dataset import BucketSampler, VideoDatasetWithResizing
    from torch.utils.data import DataLoader

    dataset_dirs = VideoDatasetWithResizing(
        data_root="assets/tests/",
        caption_column="prompts_multi.txt",
        video_column="videos_multi.txt",
        max_num_frames=49,
        id_token=None,
        random_flip=None,
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
            assert len(captions) == 8 and isinstance(captions[0], str)
            assert videos.shape == (8, 48, 3, 480, 720)
            first = True
        else:
            assert len(captions) == 8 and isinstance(captions[0], str)
            assert videos.shape == (8, 48, 3, 256, 360)
            break


if __name__ == "__main__":
    sys.path.append("./training")

    test_video_dataset()
    test_video_dataset_with_resizing()
    test_video_dataset_with_bucket_sampler()
