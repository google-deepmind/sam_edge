# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run experiments on SAM and the edge of stability using a transformer model."""
# This is a modification of haiku/examples/transformer/train.py
import argparse
import time
from typing import Any, MutableMapping

import dataset
import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jrandom
import more_tree_utils as mtu
import sam_edge
import transformer_model

# pylint: disable=bad-continuation

parser = argparse.ArgumentParser()

parser.add_argument("--training_data",
                    type=str,
                    required=True)

parser.add_argument("--test_data",
                    type=str,
                    required=True)

parser.add_argument("--batch_size",
                    type=int,
                    default=128)

parser.add_argument("--time_limit_in_hours",
                    type=float,
                    default=1.0)

parser.add_argument("--hessian_check_gap",
                    type=float,
                    help=("If not None, the number of hours "
                          + "between checks of the hessian norm"))

parser.add_argument("--num_layers",
                    type=int,
                    default=6)

parser.add_argument("--num_heads",
                    type=int,
                    default=8)

parser.add_argument("--key_size",
                    type=int,
                    default=32)

parser.add_argument("--model_size",
                    type=int,
                    default=128)

parser.add_argument("--sequence_length",
                    type=int,
                    default=64)

parser.add_argument("--step_size",
                    type=float,
                    default=0.1)

parser.add_argument("--rho",
                    type=float,
                    default=0.0,
                    help="SAM offset parameter -- if rho is 0, SAM is not used")

parser.add_argument("--eigs_curve_output",
                    type=str,
                    default="/tmp/eigs.pdf")

parser.add_argument("--eigs_se_only_output",
                    type=str,
                    default="/tmp/eigs_se_only.pdf",
                    help=("Output for plotting the eigenvalues of "
                          + "the hessian and the SAM-edge only"))

parser.add_argument("--alignment_curve_output",
                    type=str,
                    default="/tmp/a.pdf")

parser.add_argument("--loss_curve_output",
                    type=str,
                    default="/tmp/ell.pdf")

parser.add_argument("--raw_data_output",
                    type=str,
                    default="/tmp/raw.txt")

parser.add_argument("--num_principal_components",
                    type=int,
                    default=1)

args = parser.parse_args()

SEED_FACTOR = 100000
VOCAB_SIZE = 128
PAD_TOKEN = 0

_Batch = dataset.Batch
_Metrics = MutableMapping[str, Any]


def loss_fn(x, logits, y):
  """Computes the (scalar) language modelling loss on `data` w.r.t. params."""
  log_probs = jax.nn.log_softmax(logits)  # [B, T, V]
  onehot_y = jax.nn.one_hot(y, VOCAB_SIZE)
  log_likelihood = jnp.sum(onehot_y * log_probs, axis=-1)  # [B, T]

  # Loss is the average negative log-likelihood per (non-masked) token.
  mask = jnp.not_equal(x, dataset.PAD_TOKEN)  # [B, T]
  return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # []

# Create the datasets
print("Generating training data")
with open(args.training_data, mode="r") as file:
  train_dataset = dataset.load_ascii_dataset(
    corpus=file.read(),
    batch_size=args.batch_size,
    sequence_length=args.sequence_length
  )
print("Generating test data")
with open(args.test_data, mode="r") as file:
  raw_test_data = file.read()
  num_test_batches = (len(raw_test_data)//
                      (args.batch_size*args.sequence_length))
  test_dataset = dataset.load_ascii_dataset(
    corpus=raw_test_data,
    batch_size=args.batch_size,
    sequence_length=args.sequence_length
  )


def test_loss_fn(params_, lm, test_dataset_, num_test_batches_):
  """Computes the test loss."""
  @jit
  def batch_error(x, y):
    logits = lm.apply(params_, x)
    return loss_fn(x, logits, y)

  total_loss = 0.0
  num_batches = 0
  for data_ in test_dataset_:
    total_loss += batch_error(data_.inputs, data_.targets)
    num_batches += 1
    if num_batches == num_test_batches_:
      break
  return total_loss/num_batches

print("Initializing model")


def forward_pass(x):
  lm = transformer_model.LanguageModel(
    model_size=args.model_size,
    vocab_size=VOCAB_SIZE,
    pad_token=PAD_TOKEN,
    transformer=transformer_model.Transformer(
      num_heads=args.num_heads,
      num_layers=args.num_layers,
      attn_size=args.key_size,
    ))
  return lm(x)

forward_pass = hk.transform(forward_pass)
forward_pass = hk.without_apply_rng(forward_pass)
rng = jax.random.PRNGKey(int(SEED_FACTOR*time.time()) % SEED_FACTOR)
data = next(train_dataset)
rng, subkey = jrandom.split(rng)
params = forward_pass.init(subkey, data.inputs)
parameter_count = mtu.count_parameters(params)
print(f"  {parameter_count} parameters")

print("Training")
params = sam_edge.train(params,
                        forward_pass,
                        loss_fn,
                        train_dataset,
                        args.step_size,
                        args.rho,
                        args.hessian_check_gap,
                        args.eigs_curve_output,
                        args.eigs_se_only_output,
                        args.alignment_curve_output,
                        args.loss_curve_output,
                        args.raw_data_output,
                        args.num_principal_components,
                        args.time_limit_in_hours,
                        rng)

test_loss = test_loss_fn(params,
                         forward_pass,
                         test_dataset,
                         num_test_batches)
print("=======================")
print(f"test_loss: {test_loss}")
