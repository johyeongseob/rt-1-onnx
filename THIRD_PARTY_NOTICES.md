# Third-Party Notices

This project builds on third-party source code, pretrained models, datasets,
and software libraries. The Apache License 2.0 in [`LICENSE.md`](LICENSE.md)
applies to the original code and documentation authored for this project.
Third-party materials remain subject to their respective copyright notices,
licenses, and terms of use.

This project is an independent open-source conversion and validation effort.
It is not an official Google product and is not endorsed by Google.

## Google Robotics Transformer (RT-1)

- Project: Robotics Transformer (RT-1)
- Source: <https://github.com/google-research/robotics_transformer>
- License: Apache License 2.0
- Local copy: `official/robotics_transformer/`
- License copy: `official/robotics_transformer/LICENSE`

The official TensorFlow implementation is retained as the reference for model
conversion and numerical validation. The `rt1main` checkpoint is used to
generate the modular ONNX policy models. Compatibility and validation changes
made in this repository do not change the ownership or license of the upstream
work.

The official checkpoint and generated ONNX model files are not tracked in this
Git repository. Anyone downloading, converting, or redistributing those files
must preserve the applicable upstream notices and comply with the terms of the
original distribution.

## Google Tensor2Robot

- Project: Tensor2Robot
- Source: <https://github.com/google-research/tensor2robot>
- License: Apache License 2.0
- Local copy: `official/tensor2robot/`
- License copy: `official/tensor2robot/LICENSE`

Tensor2Robot is included to reproduce the official RT-1 TensorFlow environment.
This project generates its protobuf bindings and applies compatibility changes
needed by the tested TensorFlow 2.14 environment. Existing upstream copyright
and license notices are retained.

## Universal Sentence Encoder Large /5

- Model: Universal Sentence Encoder Large, version 5
- Source: <https://tfhub.dev/google/universal-sentence-encoder-large/5>
- Usage: Produces the 512-dimensional language embedding used by RT-1

The SavedModel is downloaded separately by `scripts/download_use_model.py` and
is excluded from Git. It remains subject to the license and terms published by
its model provider. This project's Apache License 2.0 does not relicense the
downloaded model or its parameters.

## Fractal RT-1 Dataset

- Dataset: `fractal20220817_data`, version `0.1.0`
- Catalog: <https://www.tensorflow.org/datasets/catalog/fractal20220817_data>
- Storage location used by the downloader:
  `gs://gresearch/robotics/fractal20220817_data/0.1.0`
- Usage: Validation of TensorFlow and ONNX inference on recorded RT-1 episodes

Dataset samples are downloaded separately by
`scripts/download_sample_episode.py` and are excluded from Git. The dataset,
images, language annotations, and recorded robot actions remain subject to the
terms published by the dataset provider. No ownership of the dataset is claimed
by this project.

When publishing results obtained from this dataset, cite the RT-1 work:

> Anthony Brohan et al. “RT-1: Robotics Transformer for Real-World Control at
> Scale.” arXiv:2212.06817, 2022. <https://arxiv.org/abs/2212.06817>

## ONNX Toolchain

The following projects are used to convert and execute the policy network:

| Component | Purpose | Upstream project | License |
| --- | --- | --- | --- |
| ONNX | Open model representation | <https://github.com/onnx/onnx> | Apache License 2.0 |
| ONNX Runtime | ONNX inference runtime | <https://github.com/microsoft/onnxruntime> | MIT License |
| tf2onnx | TensorFlow-to-ONNX conversion | <https://github.com/onnx/tensorflow-onnx> | Apache License 2.0 |

These libraries are installed as dependencies and are not relicensed by this
project. Their names and trademarks belong to their respective owners.

## Other Runtime Dependencies

This project also depends on TensorFlow, TensorFlow Probability, TF-Agents,
TensorFlow Datasets, TensorFlow Hub, NumPy, Pillow, protobuf, gRPC, and MuJoCo,
among other packages listed in `requirements.txt`. Each dependency is governed
by the license distributed with that package. Installing this project does not
alter those terms.

## Generated Artifacts

The ONNX models produced by this project contain parameters derived from the
official RT-1 checkpoint. Conversion changes the representation and runtime,
not the provenance of the underlying parameters. The original source,
copyright notices, and applicable distribution terms must be considered when
sharing generated model artifacts.

The MuJoCo visualizations produced by this project are schematic renderings of
RT-1 end-effector actions. They do not contain or reproduce the original
Everyday Robots robot CAD model.

## No Trademark License or Endorsement

Google, TensorFlow, RT-1, ONNX, ONNX Runtime, Microsoft, MuJoCo, and other names
may be trademarks of their respective owners. Their use here is solely for
identification and attribution. No sponsorship, endorsement, or trademark
license is implied.
