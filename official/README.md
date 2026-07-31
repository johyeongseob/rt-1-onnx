# RT-1 TensorFlow Setup on WSL2

## Sources

- RT-1: [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)
- Tensor2Robot: [google-research/tensor2robot](https://github.com/google-research/tensor2robot)

## 1. Install WSL2

Run PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart Windows if prompted, launch Ubuntu, and create a UNIX user.

## 2. Set up Ubuntu

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  git \
  build-essential \
  protobuf-compiler

mkdir -p ~/venvs
python3 -m venv ~/venvs/rt1-tensorflow
source ~/venvs/rt1-tensorflow/bin/activate
```

## 3. Clone the official repositories

```bash
cd /mnt/c/Users/(your_machine_name)/Desktop/rt-1-lab/official

git clone https://github.com/google-research/robotics_transformer.git
git clone https://github.com/google-research/tensor2robot.git
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip wheel

python -m pip install \
  "setuptools==80.9.0" \
  "tensorflow==2.14.0" \
  "tensorflow-probability==0.22.1" \
  "tf-agents==0.18.0" \
  "numpy==1.26.4" \
  "grpcio==1.59.3" \
  "grpcio-tools==1.59.3" \
  "protobuf==4.21.12" \
  "tensorflow-datasets==4.9.4" \
  "tensorflow-metadata==1.17.2" \
  "typing-extensions==4.5.0" \
  "cryptography==41.0.7" \
  gin-config \
  tf-slim

python -m pip check
```

## 5. Generate the Tensor2Robot protobuf module

```bash
python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  tensor2robot/proto/t2r.proto
```

## 6. Patch Tensor2Robot

Edit `tensor2robot/utils/tensorspec_utils.py`.

Replace:

```python
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest
TSPEC = contrib_framework.TensorSpec
```

With:

```python
from tensorflow.python.util import nest

TSPEC = tf.TensorSpec
```

Replace both occurrences of:

```python
isinstance(value, contrib_framework.TensorSpec)
```

With:

```python
isinstance(value, TSPEC)
```

## 7. Run the test

```bash
python -m robotics_transformer.tokenizers.action_tokenizer_test
```

Expected result:

```text
Ran 9 tests

OK (skipped=1)
```

## 8. Download one dataset sample

The script reads the public RT-1 training dataset without downloading the
complete dataset. By default, it extracts the first successful episode.

```bash
python download_sample_episode.py
```

Each sample is written to its own directory under `data/fractal_samples/`,
for example `data/fractal_samples/episode_00000/`. Downloaded samples are
excluded from Git.

To select an exact zero-based episode index:

```bash
python download_sample_episode.py --episode-index 0
```

To download an inclusive range of episodes:

```bash
python download_sample_episode.py --start-index 2 --end-index 10
```

Existing episode directories are skipped without being overwritten.

## Resume later

```powershell
wsl -d Ubuntu-22.04
```

```bash
source ~/venvs/rt1-tensorflow/bin/activate
cd /mnt/c/Users/(your_machine_name)/Desktop/rt-1-lab/official
```
