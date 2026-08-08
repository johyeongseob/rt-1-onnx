"""Restore the trained RT-1 TokenLearner from an object checkpoint."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from film_efficientnet_checkpoint import local_checkpoint_prefix
from robotics_transformer.tokenizers import token_learner


def restore_token_learner(
    model_dir: Path, features: tf.Tensor
) -> token_learner.TokenLearnerModule:
  """Build and restore the eight-token rt1main TokenLearner."""
  learner = token_learner.TokenLearnerModule(num_tokens=8)
  learner(features, training=False)

  image_tokenizer = tf.Module()
  image_tokenizer._token_learner = learner  # pylint: disable=protected-access
  actor_network = tf.Module()
  actor_network._image_tokenizer = image_tokenizer  # pylint: disable=protected-access
  agent = tf.Module()
  agent._actor_network = actor_network  # pylint: disable=protected-access

  checkpoint = tf.train.Checkpoint(agent=agent)
  status = checkpoint.restore(local_checkpoint_prefix(model_dir))
  status.expect_partial()
  status.assert_existing_objects_matched()
  return learner


__all__ = ["restore_token_learner"]
