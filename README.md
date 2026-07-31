# rt-1-lab

Converting Robotics Transformer (RT-1) from TensorFlow to ONNX.

## Download dataset samples

The downloader reads the public RT-1 training dataset from Google Cloud
Storage without downloading the complete dataset. Samples are stored under
`data/fractal_samples/` and excluded from Git.

Run the commands from the repository root in the WSL TensorFlow environment.

Download an exact zero-based episode index:

```bash
python scripts/download_sample_episode.py --episode-index 0
```

Download an inclusive range of episodes:

```bash
python scripts/download_sample_episode.py --start-index 2 --end-index 10
```

Without an episode index, the script searches for the first episode annotated
as successful within the configured search limit:

```bash
python scripts/download_sample_episode.py
```

Each episode is stored in a separate directory, for example:

```text
data/fractal_samples/episode_00000/
```

Existing episode directories are skipped without being overwritten.
