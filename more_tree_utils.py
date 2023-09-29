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

"""Misc. utilities for pytrees."""
import jax
from jax import jit
from jax import tree_util
import jax.numpy as jnp

NORMALIZING_EPS = 1e-5


@jit
def normalize(t):
  norm = get_vector_norm(t)
  return tree_util.tree_map(lambda t_leaf: t_leaf/(norm + NORMALIZING_EPS), t)


@jit
def project_out_and_normalize(s, t):
  """Project out and normalize.

  Args:
    s: pytree
    t: another pytree

  Returns:
    The pytree obtained by projecting the flattening of
    t onto the flattening of s, normalizing the result,
    and then reshaping into a pytree.
  """

  s_dot_t = get_tree_dot(s, t)
  # pylint: disable=g-long-lambda
  new_part = tree_util.tree_map(lambda s_leaf, t_leaf:
                                t_leaf - s_dot_t * s_leaf, s, t)
  return normalize(new_part)


def get_orthonormal_basis(t_list):
  k = len(t_list)
  t_list[0] = normalize(t_list[0])
  for i in range(k):
    for j in range(i+1, k):
      t_list[j] = project_out_and_normalize(t_list[i], t_list[j])
  return t_list


@jit
def get_vector_norm(t):
  squared_norms = tree_util.tree_map(lambda x: jnp.sum(x*x), t)
  return jnp.sqrt(jnp.sum(jnp.array(tree_util.tree_leaves(squared_norms))))


def count_parameters(t):
  leaf_parameter_counts = tree_util.tree_map(lambda x: x.size, t)
  return jnp.sum(jnp.array(tree_util.tree_leaves(leaf_parameter_counts)))


@jit
def get_tree_dot(s, t):
  leaf_dots = tree_util.tree_map(lambda si, ti: jnp.sum(si*ti), s, t)
  return jnp.sum(jnp.array(tree_util.tree_leaves(leaf_dots)))


@jit
def get_alignment(s, t):
  return jnp.abs(get_tree_dot(s, t))/(get_vector_norm(s)*get_vector_norm(t))


@jit
def get_random_direction(rng_key, t):
  """Sample a unit length pytree.

  Args:
    rng_key: RNG key
    t: a pytree

  Returns:
    A pytree with the same shape as t, whose
    leaves are collectively sampled uniformly
    at random from the unit ball.
  """
  def sample_at_leaf(sub_key, shape):
    return jax.random.normal(sub_key, shape)

  flat_t, treedef = tree_util.tree_flatten(t)
  leaf_shapes = tree_util.tree_map(lambda x: x.shape, flat_t)
  rng_keys = jax.random.split(rng_key, len(leaf_shapes))

  new_leaves = [sample_at_leaf(rng_keys[i], leaf_shapes[i])
                for i in range(len(leaf_shapes))]
  new_leaves = normalize(new_leaves)
  return tree_util.tree_unflatten(treedef, new_leaves)
