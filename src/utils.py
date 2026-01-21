# split the whole dataset accordingly

def split_by_video(samples, train_ratio =0.7, val_ratio = 0.15):
    video_ids = list({s["video_id"] for s in samples})
    video_ids.sort()
    # decide how many videos go into each split
    n_train = int(len(video_ids) * train_ratio)
    n_val = int(len(video_ids) * val_ratio)

    train_videos = set(video_ids[:n_train])
    val_videos = set(video_ids[n_train:n_train+ n_val])
    test_videos = set(video_ids[n_train + n_val:])

    # give all videos in a vid dictionary
    train_samples = [s for s in samples if s["video_id"] in train_videos]
    val_samples = [s for s in samples if s["video_id"] in val_videos]
    test_samples = [s for s in samples if s["video_id"] in test_videos]


    return train_samples, val_samples, test_samples