# Contributing to RT-1 ONNX

Thank you for your interest in contributing to RT-1 ONNX. This project
converts the official Google TensorFlow RT-1 policy network into modular ONNX
models and validates numerical equivalence on recorded RT-1 robot episodes.

Contributions should preserve the following principles:

- Treat the official TensorFlow RT-1 implementation as the numerical reference.
- Keep conversion and validation procedures reproducible.
- Do not trade output equivalence for performance without documenting the
  accuracy impact.
- Preserve the provenance, copyright notices, and licenses of third-party code,
  models, checkpoints, and datasets.
- Do not claim robot capabilities or physical safety properties that have not
  been validated.

## Ways to Contribute

Contributions are welcome in areas such as:

- TensorFlow-to-ONNX conversion reliability
- ONNX Runtime inference and pipeline orchestration
- Tests for preprocessing, image history, attention masks, and action decoding
- TensorFlow-ONNX equivalence validation on additional episodes
- CPU and accelerator performance measurement
- C++, OpenVINO, Docker, and edge deployment support
- MuJoCo action visualization
- Setup, troubleshooting, and usage documentation
- Error handling and diagnostics

## Reference Development Environment

The conversion and equivalence results currently documented by this project
were produced with:

- Windows with WSL2
- Ubuntu 22.04
- Python 3.10
- TensorFlow 2.14.0
- ONNX Runtime 1.18.1

Other environments are welcome, but your pull request must identify the
operating system, Python version, execution device, and relevant package
versions used for validation.

Keeping the virtual environment in the WSL Linux filesystem is recommended:

```bash
python3.10 -m venv ~/venvs/rt1-tensorflow
source ~/venvs/rt1-tensorflow/bin/activate
```

Install the pinned project dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

See [`official/README.md`](official/README.md) for WSL setup, preparation of the
official RT-1 and Tensor2Robot sources, protobuf generation, compatibility
changes, and the official tokenizer test.

## External Models and Data

Download Universal Sentence Encoder Large `/5`:

```bash
python scripts/download_use_model.py
```

Download a Fractal episode for validation:

```bash
python scripts/download_sample_episode.py --episode-index 1
```

The official `rt1main` checkpoint is expected at:

```text
official/robotics_transformer/trained_checkpoints/rt1main/
```

Models, checkpoints, datasets, and generated validation artifacts are excluded
from Git by default. If a contribution needs to include an artifact, explain
its purpose, provenance, license, size, and reproduction procedure in the pull
request.

## Development Workflow

1. Identify the issue and keep the proposed change narrowly scoped.
2. Make the code change without unrelated formatting or file movement.
3. Run the minimum validation appropriate for the affected module.
4. Run stage-level and end-to-end equivalence checks for changes that may affect
   model outputs.
5. Update documentation when commands, paths, inputs, or user-visible behavior
   change.
6. Describe the motivation, environment, validation commands, and results in the
   pull request.

Avoid combining a numerical or behavioral change with a large mechanical
rewrite. Doing so makes the source of output differences difficult to audit.

## Python Guidelines

- Follow the existing Python style and two-space indentation.
- Add docstrings to public functions and non-obvious numerical operations.
- Prefer `pathlib.Path` for filesystem paths.
- Validate input files, tensor shapes, dtypes, and value ranges explicitly.
- Document sources of randomness and set seeds when reproducibility matters.
- Raise actionable errors instead of silently ignoring invalid states.
- Keep model input and output names and shapes visible and testable.
- Do not introduce unnecessary precision changes or framework conversions.

## Validation by Change Type

### Documentation-only changes

- Confirm that commands, paths, versions, and links match the current code.
- Describe the inference runtime as ONNX-only only when the ONNX USE embedding
  and complete RT-1 action equivalence checks continue to pass.
- Do not describe the MuJoCo visualization as the original EDR robot or its
  actual joint trajectory.

### Preprocessing or image processing

```bash
python official/validation/validate_preprocessors.py
python onnx/validation/validate_preprocessors.py
python comparison/preprocessors.py

python official/validation/validate_resize.py
python onnx/validation/validate_resize.py
python comparison/resize.py
```

### FiLM-EfficientNet

```bash
python official/validation/validate_film_efficientnet.py
python onnx/validation/validate_film_efficientnet.py
python comparison/film_efficientnet.py
```

Expected output shape: `[1, 9, 9, 512]`.

### TokenLearner

```bash
python official/validation/validate_token_learner.py
python onnx/validation/validate_token_learner.py
python comparison/token_learner.py
```

Expected output shape: `[1, 8, 512]`.

### Image history

```bash
python official/validation/validate_image_history.py
python onnx/validation/validate_image_history.py
python comparison/image_history.py
```

Expected output shape: `[1, 6, 8, 512]`.

### Transformer or attention mask

```bash
python comparison/prepare_transformer_input.py
python official/validation/validate_transformer.py
python onnx/validation/validate_transformer.py
python comparison/transformer.py
```

Expected Transformer output shape: `[1, 114, 256]`.

### Action decoder

```bash
python official/validation/validate_action_decoder.py
python onnx/validation/validate_action_decoder.py
python comparison/action.py
```

Check token and continuous values for every action component:

- `terminate_episode`
- `world_vector`
- `rotation_delta`
- `gripper_closedness_action`
- `base_displacement_vector`
- `base_displacement_vertical_rotation`

### Conversion or inference pipeline

First compare the end-to-end result for the first six frames:

```bash
python official/validation/validate_end_to_end.py
python onnx/validation/validate_end_to_end.py
python comparison/end_to_end_models.py
```

Then compare all 66 frames of episode 1:

```bash
python official/validation/validate_episode.py
python onnx/validation/validate_episode.py
python comparison/end_to_end_episode.py
```

The current reference result is:

```text
Frames compared: 66
Token mismatch frames: []
Action mismatches: []
Maximum absolute action error: 2.384185791015625e-07
Match: True
```

Do not hide failures by relaxing tolerances. If a tolerance must change, include
the justification and before-and-after results in the pull request.

See [`comparison/README.md`](comparison/README.md) for the complete validation
procedure, expected tensor shapes, and tolerances.

## Performance Contributions

Performance comparisons should report at least:

- CPU, GPU, or accelerator model
- Operating system and runtime environment
- Python and inference runtime versions
- Warm-up count
- Number of measured frames or episodes
- Initial model loading time
- Mean latency and, when practical, percentile latency
- Output error and action-token agreement before and after the change

Use identical inputs and precision settings for comparisons. A speed result
without an equivalence result is not sufficient for changes to the policy
runtime.

## MuJoCo Visualization Contributions

The MuJoCo model is a schematic visualization of RT-1 end-effector actions. It
is not the original Everyday Robots model.

- Do not state that RT-1 outputs individual arm-joint commands.
- Clearly identify virtual link lengths, joint limits, and inverse-kinematics
  parameters as visualization choices.
- Do not claim to reconstruct the original robot CAD or actual joint trajectory.
- Keep visualization changes separate from the ONNX action JSON and policy
  inference results.

## Modifying Vendored Google Code

Files under `official/robotics_transformer/` and `official/tensor2robot/` are
third-party works.

- Retain all existing copyright and license notices.
- Add a clear modification notice and purpose to files you change.
- Prefer placing project-owned conversion and validation code under `onnx/`,
  `comparison/`, or `scripts/`.
- Do not reformat large upstream files unnecessarily.
- Keep the difference between upstream and modified code auditable.

Example modification notice:

```python
# Modifications Copyright 2026 rt-1-onnx contributors
# Modified for TensorFlow 2.14 compatibility and ONNX validation.
```

## Files That Should Not Be Committed

Do not commit the following by default:

- Python virtual environments
- `__pycache__` directories and `.pyc` files
- Official checkpoints
- The downloaded ONNX USE model
- Generated ONNX models
- Downloaded Fractal episodes
- Intermediate NumPy tensors and validation outputs
- General visualization artifacts
- PowerPoint temporary files

A small documentation image or representative demo may be included as an
explicit exception. Keep the `.gitignore` exception limited to the required
file, and document its source and purpose.

## Pull Request Checklist

Include the following information in a pull request:

- Problem being solved
- Summary and rationale of the change
- Affected modules
- Operating system and Python/runtime versions
- Validation commands executed
- Numerical results and tolerances
- Known limitations
- Before-and-after images for visual changes
- Provenance and license for any new external dependency or asset

If a required check could not be run, state why and identify the environment or
artifact needed to complete it.

## Reporting Issues

When reporting a bug, provide as much of the following as possible:

- Reproduction command
- Complete error message and stack trace
- Episode index and instruction
- Operating system, Python version, and relevant package versions
- Expected and actual behavior
- The stage where the issue occurs: official TensorFlow, ONNX, or comparison

Do not attach private checkpoints, full datasets, credentials, personal data,
or sensitive system paths to a public issue.

## License

Unless explicitly stated otherwise, contributions submitted to this project are
provided under the Apache License 2.0 in [`LICENSE.md`](LICENSE.md).

Third-party code, models, checkpoints, and datasets remain subject to their
respective terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

Contributors are responsible for ensuring that they have the right to submit
their code and materials under the applicable terms.
