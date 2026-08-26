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

The `rt1main` checkpoint is distributed as part of the official Robotics
Transformer repository. The repository is licensed under the Apache License
2.0, and no separate checkpoint-specific license was identified.

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
- ONNX conversion:
  <https://huggingface.co/SamLowe/universal-sentence-encoder-large-5-onnx>
- ONNX conversion license: Apache License 2.0
- Usage: Produces the 512-dimensional language embedding used by RT-1

The ONNX model is downloaded separately by `scripts/download_use_model.py` and
is excluded from Git. It uses ONNX Runtime Extensions to execute its embedded
tokenizer. The model remains subject to the license and terms published by its
original provider and ONNX converter. This project's Apache License 2.0 does
not relicense the downloaded model or its parameters.

## Fractal RT-1 Dataset

- Dataset: `fractal20220817_data`, version `0.1.0`
- Catalog: <https://www.tensorflow.org/datasets/catalog/fractal20220817_data>
- Open X-Embodiment collection and licensing notice:
  <https://github.com/google-deepmind/open_x_embodiment>
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: <https://creativecommons.org/licenses/by/4.0/>
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

## Direct Python Dependencies

The following table covers every package directly pinned in
`requirements.txt`. Versions are the versions used for the documented
conversion and validation environment.

| Component | Version | Purpose | Upstream project | License |
| --- | --- | --- | --- | --- |
| setuptools | 80.9.0 | Python package installation | <https://github.com/pypa/setuptools> | MIT |
| wheel | 0.47.0 | Python wheel support | <https://github.com/pypa/wheel> | MIT |
| TensorFlow | 2.14.0 | Official RT-1 reference and conversion | <https://github.com/tensorflow/tensorflow> | Apache-2.0 |
| TensorFlow Probability | 0.22.1 | TensorFlow probability utilities | <https://github.com/tensorflow/probability> | Apache-2.0 |
| TF-Agents | 0.18.0 | RT-1 policy dependencies | <https://github.com/tensorflow/agents> | Apache-2.0 |
| TensorFlow Datasets | 4.9.4 | Fractal dataset access | <https://github.com/tensorflow/datasets> | Apache-2.0 |
| TensorFlow Metadata | 1.17.2 | Dataset metadata types | <https://github.com/tensorflow/metadata> | Apache-2.0 |
| TF-Slim | 1.1.0 | TensorFlow model utilities | <https://github.com/google-research/tf-slim> | Apache-2.0 |
| Gin Config | 0.5.0 | RT-1 configuration parsing | <https://github.com/google/gin-config> | Apache-2.0 |
| typing-extensions | 4.5.0 | Python typing backports | <https://github.com/python/typing_extensions> | PSF-2.0 |
| cryptography | 41.0.7 | Authentication dependency support | <https://github.com/pyca/cryptography> | Apache-2.0 OR BSD-3-Clause |
| grpcio | 1.59.3 | gRPC runtime | <https://github.com/grpc/grpc> | Apache-2.0 |
| grpcio-tools | 1.59.3 | Tensor2Robot protobuf generation | <https://github.com/grpc/grpc> | Apache-2.0 |
| protobuf | 4.21.12 | Protocol Buffer runtime | <https://github.com/protocolbuffers/protobuf> | BSD-3-Clause |
| tf2onnx | 1.17.0 | TensorFlow-to-ONNX conversion | <https://github.com/onnx/tensorflow-onnx> | Apache-2.0 |
| ONNX | 1.16.2 | Open model representation | <https://github.com/onnx/onnx> | Apache-2.0 |
| ONNX Runtime | 1.18.1 | ONNX inference runtime | <https://github.com/microsoft/onnxruntime> | MIT |
| ONNX Runtime Extensions | 0.15.2 | USE tokenizer custom operators | <https://github.com/microsoft/onnxruntime-extensions> | MIT |
| NumPy | 1.26.4 | Tensor and numerical processing | <https://github.com/numpy/numpy> | BSD-3-Clause |
| Pillow | 12.3.0 | Image and GIF processing | <https://github.com/python-pillow/Pillow> | MIT-CMU |
| MuJoCo | 3.11.0 | Action visualization | <https://github.com/google-deepmind/mujoco> | Apache-2.0 |

These packages are installed as dependencies and are not vendored or
relicensed by this project. Their source distributions and binary wheels may
include transitive or bundled components under additional compatible licenses;
the license files distributed with each installed package remain authoritative.
Names and trademarks belong to their respective owners.

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
