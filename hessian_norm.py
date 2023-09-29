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

"""Compute the norm of the Hessian of a Jax model."""
import jax
from jax import grad
from jax import jit
from jax import tree_util
import jax.numpy as jnp
import more_tree_utils as mtu

# The size of the offsets used to estimate || H ||_op -- the measurement will be
# bounded above and below by the maximum of minimum of || H ||_op in a ball of
# this size centered at H
DIFFERENTIATION_STEP_SIZE = 0.001
POWER_ITERATION_EPS = 0.001  # relative error


class CurvatureEstimator:
  """Estimate the curvature of a loss function."""

  def __init__(self, loss, rng, iteration_limit=100):
    self.loss = loss
    self.loss_grad = grad(loss)
    self.rng = rng
    self.iteration_limit = iteration_limit

  def hessian_vector_product(self, params, x, y, vector):
    """Compute the product between the Hessian and a vector.

    Args:
      params: a pytree with the parameters of the model
      x: feature tensor
      y: label tensor
      vector: a pytree of the same shape as params

    Returns:
      The product of the Hessian of the loss with the vector.
    """

    @jit
    def hessian_vector_product_wo_self(params, x, y, vector):
    # doesn't take self as an input, so that it can be jitted
      eps = DIFFERENTIATION_STEP_SIZE

      plus_location = tree_util.tree_map(lambda theta, v: theta + eps*v,
                                         params,
                                         vector)
      plus_gradient = self.loss_grad(plus_location, x, y)
      minus_location = tree_util.tree_map(lambda theta, v: theta - eps*v,
                                          params,
                                          vector)
      minus_gradient = self.loss_grad(minus_location, x, y)
      # pylint: disable=g-long-lambda
      return tree_util.tree_map(lambda x, y: ((x - y)/
                                              (2.0*eps
                                               + mtu.NORMALIZING_EPS)),
                                plus_gradient,
                                minus_gradient)

    return hessian_vector_product_wo_self(params, x, y, vector)

  def curvature_and_direction(self, params, x, y):
    self.rng, subkey = jax.random.split(self.rng)
    v = mtu.get_random_direction(subkey, params)

    return self.curvature_and_direction_with_start(params, x, y, v)

  def curvature_and_direction_with_start(self, params, x, y, v):
    """Compute the curvature of the loss with respect to x and y.

    Args:
      params: a pytree with the parameters of the model
      x: feature tensor
      y: label tensor
      v: a pytree of the same shape as params, to use as a warm start

    Returns:
      The principal eigenvalue of the Hessian (which could be negative),
      and the principal eigenvector.
    """

    @jit
    def iteration(old_v):
      # perform hessian-vector product
      product = self.hessian_vector_product(params, x, y, old_v)

      # calculate the norm
      product_norm = mtu.get_vector_norm(product)

      # normalize the product to get a new iterate v
      new_v = jax.tree_util.tree_map(lambda w: w/product_norm, product)

      return new_v, product, product_norm

    last_norm = None
    current_norm = 1.0
    t = 0
    while (((last_norm is None)
            or ((abs(current_norm - last_norm)
                 > current_norm*POWER_ITERATION_EPS)
                and abs(current_norm) > POWER_ITERATION_EPS))
           and (t < self.iteration_limit)):
      last_norm = current_norm
      v, product, current_norm = iteration(v)
      t += 1

    leaf_dots_tree = tree_util.tree_map(lambda x, y: jnp.sum(x*y), v, product)
    leaf_dots = tree_util.tree_leaves(leaf_dots_tree)
    return jnp.sum(jnp.array(leaf_dots)), v

  def hessian_operator_norm(self, params, x, y):
    mu, _ = self.curvature_and_direction(params, x, y)
    return abs(mu)

  # * not vectorized (at least among the components), for now
  # * to save memory, only return the principal eigenvector
  def hessian_top_eigenvalues(self, params, x, y, k):
    """Compute the k top eigenvalues of the Hessian.

    Args:
      params: A pytree with the parameters of the model
      x: feature tensor
      y: label tensor
      k: number of eigenvalues to return

    Returns:
      A jnp array with the eigenvalues, and pytree of the same shape
      as params with the principal eigenvector.

    Note: assumes that tghe top eigenvalues are positive.
    """

    component_list = list()
    for _ in range(k):
      self.rng, subkey = jax.random.split(self.rng)
      component_list.append(mtu.get_random_direction(subkey, params))
    component_list = mtu.get_orthonormal_basis(component_list)

    @jit
    def iteration(old_components):
      products = list()
      product_norms = list()
      for i in range(k):
        # perform hessian-vector product
        products.append(self.hessian_vector_product(params, x, y,
                                                    old_components[i]))
        product_norms.append(mtu.get_vector_norm(products[i]))

      return mtu.get_orthonormal_basis(products), jnp.array(product_norms)

    last_norms = None
    current_norms = jnp.ones(k)
    t = 0
    while (((last_norms is None)
            or ((jnp.linalg.norm(current_norms - last_norms)
                 > jnp.linalg.norm(current_norms)*POWER_ITERATION_EPS)
                and jnp.linalg.norm(current_norms) > POWER_ITERATION_EPS))
           and (t < self.iteration_limit)):
      last_norms = current_norms
      component_list, current_norms = iteration(component_list)
      t += 1

    # sort in decreasing order
    indices_by_norms = jnp.argsort(-current_norms)

    return current_norms[indices_by_norms], component_list[indices_by_norms[0]]
