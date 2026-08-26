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

## 3. Verify the included official sources

The official RT-1 and Tensor2Robot source trees are already included in this
repository. Do not clone them again inside `official/`.

```bash
cd /mnt/c/Users/<username>/Desktop/rt-1-onnx

test -d official/robotics_transformer
test -d official/tensor2robot
```

The upstream source and license files are retained in each directory. Large
RT-1 checkpoints are excluded from Git and must be prepared separately.

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## 5. Generate the Tensor2Robot protobuf module

Move to the `official/` directory before generating the module.

```bash
cd /mnt/c/Users/<username>/Desktop/rt-1-onnx/official

python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  tensor2robot/proto/t2r.proto
```

## 6. TensorFlow 2.x compatibility patch

The included `tensor2robot/utils/tensorspec_utils.py` already contains the
TensorFlow 2.x compatibility patch used by this project. The original
TensorFlow 1.x implementation relied on `tensorflow.contrib`, which is not
available in TensorFlow 2.14.

The applied replacement uses:

```python
from tensorflow.python.util import nest

TSPEC = tf.TensorSpec
```

No additional manual edit is required after cloning this project.

## 7. Run the test

```bash
python -m robotics_transformer.tokenizers.action_tokenizer_test
```

Expected result:

```text
Ran 9 tests

OK (skipped=1)
```

## Resume later

```powershell
wsl -d Ubuntu-22.04
```

```bash
source ~/venvs/rt1-tensorflow/bin/activate
cd /mnt/c/Users/<username>/Desktop/rt-1-onnx/official
```
