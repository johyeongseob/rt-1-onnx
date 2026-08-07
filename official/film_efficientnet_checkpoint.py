"""Restore the trained RT-1 FiLM-EfficientNet from an object checkpoint."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from robotics_transformer.film_efficientnet import pretrained_efficientnet_encoder


def local_checkpoint_prefix(model_dir: Path) -> str:
  """Return the local checkpoint prefix, ignoring Google's saved CNS path."""
  checkpoint_indexes = sorted(model_dir.glob("ckpt-*.index"))
  if not checkpoint_indexes:
    raise FileNotFoundError(f"Training checkpoint was not found: {model_dir}")
  return str(checkpoint_indexes[-1])[:-len(".index")]


def restore_encoder(
    model_dir: Path, image: tf.Tensor, context: tf.Tensor
) -> pretrained_efficientnet_encoder.EfficientNetEncoder:
  """Build and restore the rt1main FiLM-EfficientNet encoder."""
  encoder = pretrained_efficientnet_encoder.EfficientNetEncoder(
      model_variant="b3",
      early_film=True,
      weights=None,
      include_top=False,
      pooling=False,
  )
  encoder(image, context=context, training=False)

  image_tokenizer = tf.Module()
  image_tokenizer._tokenizer = encoder  # pylint: disable=protected-access
  actor_network = tf.Module()
  actor_network._image_tokenizer = image_tokenizer  # pylint: disable=protected-access
  agent = tf.Module()
  agent._actor_network = actor_network  # pylint: disable=protected-access

  checkpoint = tf.train.Checkpoint(agent=agent)
  status = checkpoint.restore(local_checkpoint_prefix(model_dir))
  status.expect_partial()
  status.assert_existing_objects_matched()
  return encoder


__all__ = ["local_checkpoint_prefix", "restore_encoder"]
