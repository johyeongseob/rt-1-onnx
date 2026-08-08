"""Restore the trained RT-1 Transformer from an object checkpoint."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from film_efficientnet_checkpoint import local_checkpoint_prefix
from robotics_transformer import transformer


def restore_transformer(
    model_dir: Path,
    sequence: tf.Tensor,
    attention_mask: tf.Tensor,
) -> transformer.Transformer:
  """Build and restore the rt1main eight-layer Transformer."""
  decoder = transformer.Transformer(
      num_layers=8,
      layer_size=128,
      num_heads=8,
      feed_forward_size=512,
      dropout_rate=0.1,
      vocab_size=256,
      return_attention_scores=False,
  )
  decoder(sequence, training=False, attention_mask=attention_mask)

  actor_network = tf.Module()
  actor_network._transformer = decoder  # pylint: disable=protected-access
  agent = tf.Module()
  agent._actor_network = actor_network  # pylint: disable=protected-access

  checkpoint = tf.train.Checkpoint(agent=agent)
  status = checkpoint.restore(local_checkpoint_prefix(model_dir))
  status.expect_partial()
  status.assert_existing_objects_matched()
  return decoder


__all__ = ["restore_transformer"]
