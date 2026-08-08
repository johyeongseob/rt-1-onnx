# TensorFlow-ONNX comparison

This directory contains the cross-framework comparisons used to verify that
the modular ONNX RT-1 preserves the official TensorFlow `rt1main` behavior.
Run all commands from the repository root with the WSL environment activated.

## Validation pattern

Each stage follows the same pattern:

```text
official TensorFlow output -> validation_artifacts/.../tensorflow.*
ONNX output               -> validation_artifacts/.../onnx.*
comparison                -> Match: True
```

## Preprocessing

```bash
python official/validation/validate_preprocessors.py
python onnx/validation/validate_preprocessors.py
python comparison/preprocessors.py

python official/validation/validate_resize.py
python onnx/validation/validate_resize.py
python comparison/resize.py
```

## Language embedding

Embed `close middle drawer` with USE Large `/5` and compare it with the
embedding stored in episode 1:

```bash
python official/validation/validate_use_embedding.py
```

Small floating-point differences are expected. The tested result matches with
`rtol=1e-5` and `atol=1e-6`.

## FiLM-EfficientNet

```bash
python official/validation/validate_film_efficientnet.py
python onnx/validation/validate_film_efficientnet.py
python comparison/film_efficientnet.py
```

Expected shape: `[1, 9, 9, 512]`.

## TokenLearner

```bash
python official/validation/validate_token_learner.py
python onnx/validation/validate_token_learner.py
python comparison/token_learner.py
```

Expected shape: `[1, 8, 512]`.

## Six-frame image history

```bash
python official/validation/validate_image_history.py
python onnx/validation/validate_image_history.py
python comparison/image_history.py
```

Expected shape: `[1, 6, 8, 512]`.

## Transformer

Create the 114-token sequence and causal attention mask, then compare the
TensorFlow and ONNX transformer outputs:

```bash
python comparison/prepare_transformer_input.py
python official/validation/validate_transformer.py
python onnx/validation/validate_transformer.py
python comparison/transformer.py
```

Expected output shape: `[1, 114, 256]`.

To explicitly create the sequence from the ONNX image history:

```bash
python comparison/prepare_transformer_input.py --source onnx
```

## Action decoding

```bash
python official/validation/validate_action_decoder.py
python onnx/validation/validate_action_decoder.py
python comparison/action.py
```

The decoder produces:

- `terminate_episode`
- `world_vector` (`x`, `y`, `z`)
- `rotation_delta` (`roll`, `pitch`, `yaw`)
- `gripper_closedness_action`
- `base_displacement_vector` (`x`, `y`)
- `base_displacement_vertical_rotation` (`yaw`)

## End-to-end: first six frames

Run the official SavedModel policy and the connected ONNX pipeline, then
compare their final action:

```bash
python official/validation/validate_end_to_end.py
python onnx/validation/validate_end_to_end.py
python comparison/end_to_end_models.py
```

For reproducible comparison, omit `--instruction` so both paths use the
episode instruction. The ONNX entry point encodes the metadata instruction
with the local USE Large `/5` SavedModel rather than loading the episode's
precomputed `language_embedding.npy`.

## End-to-end: complete episode

Run all 66 frames of episode 1 through each pipeline and compare every action
in the generated JSON files:

```bash
python official/validation/validate_episode.py
python onnx/validation/validate_episode.py
python comparison/end_to_end_episode.py
```

Expected result:

```text
Frames compared: 66
Token mismatch frames: []
Action mismatches: []
Maximum absolute action error: 2.384185791015625e-07
Match: True
```

Outputs:

```text
validation_artifacts/episode_00001/episode/official.json
validation_artifacts/episode_00001/episode/onnx.json
```
