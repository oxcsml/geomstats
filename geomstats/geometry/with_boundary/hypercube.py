"""Euclidean space."""
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import EuclideanMetric, Euclidean
import jax
import jax.numpy as gs

jnp = gs

import cvxpy as cp
import jax.experimental.host_callback as hcb


def coord_dist(x):
    return gs.min(gs.stack([x, 1 - x], axis=-1), axis=-1)


def diff_coord_dist(x):
    return -1 + 2 * (x < 0.5)


def belongs(point):
    return gs.all(gs.bitwise_and(point > 0, point < 1), axis=-1)


def log_gaussian(x, sigma):
    return gs.log(1 / (gs.sqrt(2 * gs.pi) * sigma)) - (x**2 / (2 * sigma**2))


def reflected_heat_kernel(x, mean, sigma, n_max):
    n = n_max // 2
    offsets = (
        (jnp.linspace(1 + (n % 2), n_max + (n % 2), n_max) // 2) - ((n_max + 1) // 4)
    ) * 2
    x_mul = (
        jnp.linspace(1 + (n_max // 2 % 2), n_max + (n_max // 2 % 2), n_max) % 2 * 2 - 1
    )
    x = offsets + x * x_mul
    return jax.vmap(lambda x: jnp.exp(log_gaussian(x - mean, sigma)))(x).sum()


def eigen_heat_kernel(x, mean, sigma, n_max):
    ks = jnp.linspace(1, n_max, n_max)
    return (
        1
        + (
            2
            * jnp.exp(-((ks * jnp.pi * sigma) ** 2) / 2)
            * jnp.cos(ks * jnp.pi * mean)
            * jnp.cos(ks * jnp.pi * x)
        ).sum()
    )


def heat_kernel(x, mean, sigma, n_max=5, threshold=0.5):
    return jnp.where(
        sigma < threshold,
        reflected_heat_kernel(x, mean, sigma, n_max),
        eigen_heat_kernel(x, mean, sigma, n_max),
    )


def grad_log_heat_kernel(x, mean, sigma, n_max=5, threshold=0.5):
    return jnp.where(
        sigma < threshold,
        jax.grad(lambda x: jnp.log(reflected_heat_kernel(x, mean, sigma, n_max)))(x),
        jax.grad(lambda x: jnp.log(eigen_heat_kernel(x, mean, sigma, n_max)))(x),
    )


class Hypercube(Manifold):
    def __init__(self, dim, metric_type="reflected"):
        if metric_type == "Reflected":
            metric = ReflectedHypercubeMetric(dim)
        elif metric_type == "Rejection":
            metric = RejectionHypercubeMetric(dim)
        elif metric_type == "Hessian":
            metric = HessianHypercubeMetric(dim)
        else:
            raise NotImplementedError()

        super(Hypercube, self).__init__(dim, metric)

    @property
    def log_volume(self):
        return 1.0

    def random_uniform(self, state, n_samples=1):
        return jax.random.uniform(state, (n_samples, self.dim))

    def random_normal_tangent(self, state, base_point, n_samples=1):
        return state, jax.random.normal(state, (n_samples, self.dim))

    def belongs(self, point, atol=1e-12):
        return belongs(point)

    def is_tangent(self, x):
        return True

    def to_tangent(self, vector, base_point):
        return vector

    def random_point(self, rng):
        return self.random_uniform(rng)

    def distance_to_boundary(self, x):
        return jnp.sqrt((jax.vmap(jax.vmap(coord_dist))(x) ** 2).sum(axis=-1))

    @property
    def identity(self):
        return 0.5 * gs.ones(self.dim)

    def heat_kernel(self, x0, x, t):
        return jax.vmap(
            lambda x, x0, t: jax.vmap(heat_kernel, in_axes=[0, 0, None])(
                x, x0, t
            ).prod()
        )(x, x0, t)

    def grad_marginal_log_prob(self, x0, x, t):
        return jax.vmap(jax.vmap(grad_log_heat_kernel, in_axes=(0, 0, None)))(x, x0, t)


class ReflectedHypercubeMetric(EuclideanMetric):
    def __init__(self, dim, default_point_type="vector"):
        super().__init__(dim, default_point_type)

    def exp(self, tangent_vec, base_point, **kwargs):
        def exp_1d(x, tv):
            x = x + tv
            flips, x = gs.divmod(x, 1.0)
            flips = gs.mod(flips, 2)
            x = x * (1 - flips) + flips * (1 - x)

            return x

        return jax.vmap(jax.vmap(exp_1d))(base_point, tangent_vec)


class HessianHypercubeMetric(EuclideanMetric):
    def __init__(self, dim, default_point_type="vector", eps=1e-6):
        super().__init__(dim, default_point_type)
        self.eps = eps

    def metric_matrix(self, x):
        dists = jax.vmap(jax.vmap(coord_dist))(x)
        return 1 / dists**2

    def metric_inverse_matrix_sqrt(self, x):
        dists = jax.vmap(jax.vmap(coord_dist))(x)
        return dists

    def metric_inverse_matrix(self, x):
        dists = jax.vmap(jax.vmap(coord_dist))(x)
        return dists**2

    def div_metric_inverse_matrix(self, x):
        return jax.vmap(jax.vmap(lambda x: 2 * coord_dist(x) * diff_coord_dist(x)))(x)

    def lambda_x(self, x):
        # return -1 / 2 * gs.linalg.slogdet(self.metric_matrix(x))[1]
        return 1.0

    def grad_logdet_metric_matrix(self, x):
        return jax.grad(self.lambda_x)(x)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Use a retraction instead of the true exponential map."""
        # new_point = base_point + tangent_vec
        # new_point = new_point.clip(0, 1)
        # return new_point
        x = base_point
        tv = tangent_vec

        c1 = gs.log(x)
        c2 = -gs.log(-4 * (x - 1))

        t_half_1 = gs.log(0.5) - c1
        t_half_2 = gs.log(0.5) - c2

        c_half_1 = gs.log(0.5)
        c_half_2 = -gs.log(-4 * (0.5 - 1))

        no_change_exp_1 = gs.exp(c1 + tv)
        change_exp_1 = 1 - 1 / 4 * gs.exp(-c_half_2 - (tv - t_half_1))

        no_change_exp_2 = 1 - 1 / 4 * gs.exp(-c2 - tv)
        change_exp_2 = gs.exp(c_half_1 + (tv - t_half_2))

        xgth_tv_neg = gs.where(tv > t_half_2, no_change_exp_2, change_exp_2)
        xlth_tv_neg = gs.exp(c1 + tv)

        xlth_tv_pos = gs.where(tv < t_half_1, no_change_exp_1, change_exp_1)
        xgth_tv_pos = 1 - 1 / 4 * gs.exp(-c2 - (tv))

        tv_neg = gs.where(x < 0.5, xlth_tv_neg, xgth_tv_neg)
        tv_pos = gs.where(x < 0.5, xlth_tv_pos, xgth_tv_pos)

        return gs.where(tv > 0, tv_pos, tv_neg).clip(self.eps, 1 - self.eps)

    def norm(self, vector, base_point=None):
        return gs.linalg.norm(vector, axis=-1)

    def squared_norm(self, vector, base_point=None):
        return self.norm(vector, base_point=base_point) ** 2


class RejectionHypercubeMetric(EuclideanMetric):
    def __init__(self, dim, default_point_type="vector"):
        super().__init__(dim, default_point_type)

    def exp(self, tangent_vec, base_point, **kwargs):
        new_point = base_point + tangent_vec
        in_bounds = belongs(new_point).astype(float)
        return gs.where(in_bounds[:, None], new_point, base_point)
