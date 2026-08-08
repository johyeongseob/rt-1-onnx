# rt-1-lab

Convert the official TensorFlow RT-1 checkpoint to modular ONNX models and
verify that both implementations produce the same actions.


## Pipeline

```text
camera frames + 512-dimensional language embedding
                         |
                         v
        image conversion and 300 x 300 resize
                         |
                         v
                 FiLM-EfficientNet
                  [B, 9, 9, 512]
                         |
                         v
                    TokenLearner
                    [B, 8, 512]
                         |
                         v
              six-frame image history
                   [B, 6, 8, 512]
                         |
                         v
   Transformer: 6 x (8 image + 11 action slots)
                   [B, 114, 256]
                         |
                         v
          11 action tokens and detokenization
                         |
                         v
 arm xyz/rpy/gripper + base xy/yaw + mode switch
```

The ONNX implementation is intentionally modular:

- `film_efficientnet.onnx`
- `token_learner.onnx`
- `transformer.onnx`
- NumPy/Python preprocessing, pipeline orchestration, and action decoding

The weights are embedded in the ONNX files. The three ONNX models use about
182 MiB in total; the Universal Sentence Encoder is stored separately.

## Environment

The tested environment is:

- Windows with WSL2
- Ubuntu 22.04
- Python 3.10
- TensorFlow 2.14.0
- ONNX Runtime 1.18.1
- MuJoCo 3.11.0

Source code, datasets, checkpoints, ONNX models, and validation outputs remain
in the Windows repository. The Python virtual environment is stored inside
WSL at `~/venvs/rt1-tensorflow`.

See [`official/README.md`](official/README.md) for the WSL setup, official
repository clones, Tensor2Robot protobuf generation, compatibility patch, and
official tokenizer test.

Activate the environment and enter the repository:

```bash
source ~/venvs/rt1-tensorflow/bin/activate
cd /mnt/c/Users/(your_machine_name)/Desktop/rt-1-lab
```

Install the complete TensorFlow, ONNX, and visualization environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Expected official source and checkpoint paths:

```text
official/robotics_transformer/
official/tensor2robot/
official/robotics_transformer/trained_checkpoints/rt1main/
```

The two upstream `requirements.txt` files are retained unchanged. This project
uses the version-pinned root [`requirements.txt`](requirements.txt).

## Download the language encoder

RT-1 uses Universal Sentence Encoder Large `/5` to produce a 512-dimensional
instruction embedding. Download and preserve a local SavedModel copy:

```bash
python scripts/download_use_model.py
```

Output:

```text
models/universal_sentence_encoder_large/5/
```

The model directory is excluded from Git.

## Download Fractal dataset samples

The downloader reads the public RT-1 Fractal dataset directly from Google
Cloud Storage without downloading the complete dataset:

```text
gs://gresearch/robotics/fractal20220817_data/0.1.0
```

Download one exact zero-based episode index:

```bash
python scripts/download_sample_episode.py --episode-index 1
```

Download an inclusive range:

```bash
python scripts/download_sample_episode.py --start-index 2 --end-index 10
```

Without an index, the script searches for the first episode annotated as
successful within the configured search limit:

```bash
python scripts/download_sample_episode.py
```

Each episode is stored separately:

```text
data/fractal_samples/episode_00001/
|-- frames/frame_0000.png
|-- language_embedding.npy
|-- metadata.json
`-- steps.json
```

Existing episode directories are skipped rather than overwritten. Downloaded
data is excluded from Git.

## Convert the checkpoint to ONNX

Run the converters from the repository root in this order:

```bash
python onnx/scripts/convert_film_efficientnet.py
python onnx/scripts/convert_token_learner.py
python onnx/scripts/convert_transformer.py
```

Outputs:

```text
models/film_efficientnet/film_efficientnet.onnx
models/token_learner/token_learner.onnx
models/transformer/transformer.onnx
```

The converters restore the official `rt1main` weights before exporting. The
generated models are excluded from Git.

## Run ONNX inference

Run all frames of a downloaded episode with a user-provided instruction:

```bash
python onnx/validation/validate_episode.py \
  --episode-index 1 \
  --instruction "close middle drawer"
```

The instruction is encoded with the pinned local USE Large `/5` SavedModel.
FiLM-EfficientNet, TokenLearner, and Transformer then run in ONNX Runtime. The
result JSON is written to:

```text
validation_artifacts/episode_00001/episode/onnx.json
```

## TensorFlow-ONNX comparison

The conversion is verified module by module and over a complete episode. See
[`comparison/README.md`](comparison/README.md) for all validation commands,
expected tensor shapes, tolerances, and end-to-end results.

## MuJoCo visualization

Install dependencies through the root `requirements.txt`. Confirm MuJoCo and
WSLg GUI support with:

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m mujoco.viewer
```

### Live position viewer

```bash
python visualization/mujoco/visualize_world_vector.py
```

The viewer accumulates the normalized RT-1 `world_vector` deltas. This is an
action visualization, not a reconstruction of the original EDR robot or its
joint trajectory.

### Camera GIF

Create a 12 FPS camera GIF with four display frames per source frame:

```bash
python visualization/export_episode_frames_gif.py
```

Output:

```text
visualization_artifacts/episode_00001/camera_frames.gif
```

### MuJoCo gripper GIF

Create a synchronized 12 FPS MuJoCo GIF:

```bash
python visualization/mujoco/export_world_vector_gif.py
```

The simplified V-shaped gripper visualizes seven arm-action dimensions:

- cumulative `x`, `y`, `z`
- cumulative `roll`, `pitch`, `yaw`
- gripper open/close

The gripper is schematic and is not the original Everyday Robots CAD model.
The base-action visualization is intentionally omitted for episode 1 because
all 66 base-action outputs are the same center-bin value, representing an
effectively stationary base.

Output:

```text
visualization_artifacts/episode_00001/world_vector.gif
```

### Side-by-side comparison

Combine the synchronized camera and gripper GIFs into one row:

```bash
python visualization/combine_camera_vector_gifs.py
```

Output:

```text
visualization_artifacts/episode_00001/camera_and_world_vector.gif
```

## Repository layout

```text
rt-1-lab/
|-- official/               # Upstream TensorFlow code and TF validators
|-- onnx/                   # ONNX converters, runtime pipeline, validators
|-- scripts/                # Model and dataset downloaders
|-- comparison/             # Flat cross-framework comparison scripts
|-- visualization/          # Camera and MuJoCo visualization scripts
|-- data/                    # Downloaded episode samples (ignored)
|-- models/                  # USE and ONNX models (ignored)
|-- validation_artifacts/   # Intermediate arrays and JSON (ignored)
|-- visualization_artifacts/ # Generated GIFs (ignored)
`-- requirements.txt        # Pinned full-project environment
```

## Sources

- [RT-1 paper](https://arxiv.org/abs/2212.06817)
- [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)
- [google-research/tensor2robot](https://github.com/google-research/tensor2robot)
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/5)
- [MuJoCo](https://mujoco.org/)

The upstream Google repositories retain their original license and copyright
notices.
