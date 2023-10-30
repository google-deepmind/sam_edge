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

"""Perform image classification experiments regarding SAM and the edge of stability."""
# Modification from haiku/examples/transformer/train.py
import argparse
import time
from typing import Any, Sequence

from flax import linen as nn
from jax import jit
import jax.numpy as jnp
import jax.random as jrandom
import more_tree_utils as mtu
import optax
import sam_edge
import tensorflow as tf
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()

parser.add_argument("--time_limit_in_hours",
                    type=float,
                    default=4.0)

parser.add_argument("--hessian_check_gap",
                    type=float,
                    default=0.25,
                    help=("If not None, the number of seconds "
                          + "between hessian-norm measurements"))

parser.add_argument("--step_size",
                    type=float,
                    default=0.01)

parser.add_argument("--batch_size",
                    type=int,
                    default=60000)

parser.add_argument("--nn_architecture",
                    type=str,
                    default="MLP",
                    choices=["CNN", "MLP"])

parser.add_argument("--dataset",
                    type=str,
                    default="mnist",
                    choices=["mnist", "cifar10"])

parser.add_argument("--mlp_depth",
                    type=int,
                    default=4)

parser.add_argument("--mlp_width",
                    type=int,
                    default=1000)

parser.add_argument("--cnn_num_blocks",
                    type=int,
                    default=2)

parser.add_argument("--cnn_layers_per_block",
                    type=int,
                    default=1)

parser.add_argument("--cnn_feature_multiplier",
                    type=int,
                    default=16)

parser.add_argument("--mini_training_set_num_batches",
                    type=int,
                    default=None,
                    help=("If this is not None, it is the number "
                          + "of batches in a reduced training set"))

parser.add_argument("--mini_test_set_num_batches",
                    type=int,
                    default=None,
                    help=("If this is not None, it is the number "
                          +"of batches in a reduced test set"))

parser.add_argument("--rho",
                    type=float,
                    default=0.0,
                    help=("The parameter rho for SAM -- "
                          + "if rho is 0, SAM is not used"))

parser.add_argument("--eigs_curve_output",
                    type=str,
                    default="/tmp/eigs.pdf")

parser.add_argument("--eigs_se_only_output",
                    type=str,
                    default="/tmp/eigs_se_only.pdf",
                    help=("Output for plotting the eigenvalues "
                          + "of the hessian and the SAM-edge only"))

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
SHUFFLE_BATCH_SIZE = 1024

ModuleDef = Any


class CNN(nn.Module):
  """A standard convolutional neural network."""
  conv_defs: Sequence[ModuleDef]
  dense_def: ModuleDef

  @nn.compact
  def __call__(self, x):
    h = x
    for block_conv_defs in self.conv_defs:
      for conv_def in block_conv_defs:
        h = conv_def(h)
        h = nn.relu(h)
        h = nn.LayerNorm()(h)
      h = nn.max_pool(h, window_shape=(2, 2), strides=(2, 2))
    h = h.reshape((h.shape[0], -1))  # flatten
    h = self.dense_def(h)
    return h


class MLP(nn.Module):
  """A standard multi-layer perceptron."""
  input_to_hidden_def: ModuleDef
  hidden_to_hidden_defs: ModuleDef
  output_def: ModuleDef

  @nn.compact
  def __call__(self, x):
    h = x.reshape((x.shape[0], -1))  # flatten
    h = self.input_to_hidden_def(h)
    h = nn.relu(h)
    for ell in range(args.mlp_depth-2):
      h = (self.hidden_to_hidden_defs[ell])(h)
      h = nn.relu(h)
    h = self.output_def(h)
    return h


def test_error_fn(params_, model_, batches):
  """Compute test error.

  Args:
    params_: parameters of the model
    model_: the model
    batches: test data

  Returns:
    Test error.
  """

  @jit
  # targets are not one-hot encoded in the test data
  def batch_error(images, targets):
    predicted_class = jnp.argmax(model_.apply(params_, images), axis=1)
    return jnp.mean(predicted_class != targets)

  sum_batch_errors = 0.0
  num_batches = 0
  for x, y in batches:
    sum_batch_errors += batch_error(x, y)
    num_batches += 1
  return sum_batch_errors/num_batches

# ================================
# Start of main routine
print("running")

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")

data_dir = "/tmp/" + args.dataset
if args.dataset == "mnist":
  num_training_examples = 60000
  num_test_examples = 10000
  image_size = 28
  num_channels = 1
  num_classes = 10
elif args.dataset == "cifar10":
  num_training_examples = 50000
  num_test_examples = 10000
  image_size = 32
  num_channels = 3
  num_classes = 10

rng = jrandom.PRNGKey(int(SEED_FACTOR*time.time()) % SEED_FACTOR)
gn = nn.initializers.glorot_normal()
if args.nn_architecture == "CNN":
  # pylint: disable=g-complex-comprehension
  model = CNN(conv_defs=[[nn.Conv(features=(2**i*args.cnn_feature_multiplier),
                                  kernel_size=(3, 3),
                                  kernel_init=gn,
                                  padding="SAME")
                          for j in range(args.cnn_layers_per_block)]
                         for i in range(args.cnn_num_blocks)],
              dense_def=nn.Dense(features=num_classes,
                                 kernel_init=gn)
              )
  num_linear_layers = args.cnn_layers_per_block * args.cnn_num_blocks + 1
else:
  model = MLP(input_to_hidden_def=nn.Dense(features=args.mlp_width,
                                           kernel_init=gn),
              hidden_to_hidden_defs = [nn.Dense(features=args.mlp_width,
                                                kernel_init=nn.initializers.glorot_normal())
                                       for _ in range(args.mlp_depth-2)],
              output_def=nn.Dense(features=num_classes,
                                  kernel_init=gn))
  num_linear_layers = args.mlp_depth
rng, subkey = jrandom.split(rng)
params = model.init(subkey,
                    jnp.ones([args.batch_size,
                              image_size,
                              image_size,
                              num_channels]))
print("parameter count: {}".format(mtu.count_parameters(params)))


def get_train_batches():
  """Get training data."""
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name=args.dataset,
                 split="train",
                 as_supervised=True,
                 data_dir=data_dir)
  if args.mini_training_set_num_batches:
    ds = ds.take(args.mini_training_set_num_batches*args.batch_size)
  # pylint: disable=g-long-lambda
  ds = ds.map(lambda x, y:
              (tf.cast(x, dtype=tf.float32)/256.0, tf.one_hot(y, num_classes)))
  ds = ds.batch(args.batch_size, drop_remainder=True).repeat()
  return tfds.as_numpy(ds)


def get_test_batches(num_available_test_examples):
  """Get test data."""
  ds = tfds.load(name=args.dataset,
                 split="test",
                 as_supervised=True,
                 data_dir=data_dir)
  if args.mini_test_set_num_batches:
    reduced_num_test_examples = (args.mini_test_set_num_batches
                                 *args.batch_size)
    ds = ds.take(num_test_examples)
  else:
    reduced_num_test_examples = num_available_test_examples
  ds = ds.map(lambda x, y: (tf.cast(x, dtype=tf.float32)/256.0, y))
  batch_size = min(reduced_num_test_examples, args.batch_size)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)


# pylint: disable=unused-argument
def loss(x, logits, y):
  return optax.squared_error(logits, y)

train_batches = get_train_batches()
test_batches = get_test_batches(num_test_examples)

# center the data
u, _ = next(iter(train_batches))
mu = jnp.mean(u, axis=0, keepdims=True)
train_batches = ((x - mu, y) for (x, y) in train_batches)
test_batches = ((x - mu, y) for (x, y) in test_batches)


params = sam_edge.train(params,
                        model,
                        loss,
                        train_batches,
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

test_err = test_error_fn(params, model, test_batches)
print("==============")
print("Test error: {}".format(test_err))

