"""Euclidean space."""
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import EuclideanMetric, Euclidean
import jax
import numpy as np
import jax.numpy as gs
import geomstats.backend as bs

from scipy.optimize import linprog

from diffrax.misc import bounded_while_loop

import cvxpy as cp
import jax.experimental.host_callback as hcb

def proj(inp):
    A, b, base_point = inp
    X = cp.Variable(base_point.shape)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(X - base_point)),
        [A @ X.T <= b[:, None]]
    )
    problem.solve()
    return X.value

def device_proj(A, b, x):
    return hcb.call(
        proj, (A, b, x),
        result_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )

# todo: if b is always either M, 1 or M, N then
# you dont need to expand dims and you can handle
# different b with the same T matrix: this is nice
# because in our setting the T matrix is always the
# same; all that changes is the b values!


def stable_div(num, den, eps=1e-10):
    return (
        gs.sign(num)
        * gs.sign(den)
        * gs.exp(gs.log(gs.abs(num) + eps) - gs.log(gs.abs(den) + eps))
    )


def reflect(r, sn, T, b, eps=1e-6, eps2=1e-8, max_val=1e10, max_iter=100_000):
    """
    Given a set of N vectors rp in a d-polytope compute the
    set of steps rp + s, where we reflect in the direction
    normal to the face whenever we would hit a face.
    This allows computing a reflected brownian motion in
    a polytope.
    rp : ndarray : (N, d)
        the set of initial positions
    s : ndarray : (N, d)
        the set of steps to take
    T : ndarray : (M, d) and b : ndarray : M
        the matrix/vector defining the polytope by the
        inequality constraint: T x <= b
    """

    # normalize z to a direction and a
    # magnitude; we will use the magnitudes
    # here to continue reflecting until we
    # have traveled the whole distance
    def reflect_cond(val):
        rp, s, sr = val
        return gs.any(sr > 0)

    def reflect_body(val, _):
        # compute the amount we can scale in the
        # direction s before hitting any face,
        # for any of the rp, s vector pairs
        rp, s, sr = val
        sr_mask = sr > 0
        num, den = T @ rp.T - b[:, None], T @ s.T
        scale = -stable_div(num, den) * sr_mask
        scale = gs.clip(scale, -max_val, max_val)
        # we are moving in the "positive" direction
        # of s here, so mask out negative values
        scale_mask = scale <= 0
        masked_scale = scale_mask * max_val + (1 - scale_mask) * scale
        # compute the face we will hit first,
        # e.g. the minimum scaling that lands us
        # on a face
        a_argmax = masked_scale.argmin(axis=0)
        a_max = scale[a_argmax, gs.arange(scale.shape[1])]
        # us either the remaining magnitude sr
        # or the maximum scaling that lands us
        # on a face to scale in the direction s
        # add this to values of rp which we still
        # have magnitude left in their step length
        a = gs.maximum(gs.minimum(sr, a_max), 0)
        rp = rp + a[:, None] * s
        diff = T @ rp.T - b[:, None]
        idx = diff >= -eps2
        rp = rp + (T.T @ (-(gs.abs(diff) + eps) * idx)).T
        # this is just a test to ensure we are
        # still in the polytope
        # assert(gs.all(T @ rp.T <= b[:, None]))
        # we are going to reflect around the face
        # we land on, so we grab that face from T
        # and normalize it
        n = T[a_argmax, :]
        n = n / gs.sqrt(gs.sum(n**2, axis=-1))[:, None]
        # this is the reflection: note we only
        # need to reflect the direction vector s
        # about the face. for a single vector
        # and a single face we can do that using
        # this eqn: r = s - 2 * dot(s, n) * n
        # where r is the reflection. for the
        # vectorized case we compute the row-wise
        # dot products using gs.sum(s * n, axis=-1)
        s = s - (2 * gs.sum(s * n, axis=-1)[:, None] * n)
        # because n and s are normalized the
        # resulting s should be normalized too
        # we renormalize for numberical stabilty
        s = s / gs.sqrt(gs.sum(s**2, axis=-1))[:, None]
        # now we subtract the distance we
        # reflected from the magnitude, once this
        # is negative we stop reflecting that
        # vector
        sr = sr - a
        return rp, s, sr

    sr = gs.sqrt(gs.sum(sn**2, axis=-1))
    sn = sn / sr[:, None]
    rp, s, sr = bounded_while_loop(reflect_cond, reflect_body, (r, sn, sr), max_iter)

    return rp


class Polytope(Manifold):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(
        self, T=None, b=None, npz=None, metric=None, proj_type=None, metric_type="Reflected", **kwargs
    ):
        if npz is not None:
            data = np.load(npz)
            self.T, self.b = gs.array(data["T"]), gs.array(data["b"])
            if "cube" in npz:
                proj_type = "cube"
            elif "dirichlet" in npz:
                proj_type = "triangle"
        elif T is not None and b is not None:
            self.T, self.b = gs.array(T), gs.array(b)
        else:
            raise ValueError(
                "You need either the inequality matrices or "
                "an archive pointing to them"
            )
        dim = self.T.shape[1]
        if metric is None:
            if metric_type == "Reflected":
                metric = ReflectedPolytopeMetric(self.T, self.b)
            elif metric_type == "Hessian":
                if proj_type == "cube":
                    metric = HessianCubeMetric(self.T, self.b)
                    print("Using cube metric")
                elif proj_type == "triangle":
                    metric = HessianTriangleMetric(self.T, self.b)
                    print("Using triangle metric")
                else:
                    metric = HessianPolytopeMetric(self.T, self.b)
            else:
                raise NotImplementedError

        super(Polytope, self).__init__(dim=dim, metric=metric)
        self.metric = metric
        # used to compute a point in the interior of the polytope
        # which we can do random walks from to generate random samples
        xc = cp.Variable(self.T.shape[1])
        r = cp.Variable()

        problem = cp.Problem(
            cp.Maximize(r),
            [self.T @ xc.T + r * cp.norm(self.T, axis=1) <= self.b, r >= 0]
        )
        problem.solve()

        self.center = xc.value

    def exp(self, tangent_vec, base_point=None):
        return self.metric.exp(tangent_vec, base_point)

    @property
    def log_volume(self):
        # TODO: this could be computed using random uniform samples for the
        # volume of the polytope. Unclear how to implement for hessian/logbarrier.
        return self.dim

    def random_uniform(self, state, n_samples=1, step_size=1.0, num_steps=10_000):
        def walk(_, carry):
            rng, pos = carry
            rng, next_rng = jax.random.split(rng)
            samples = jax.random.normal(rng, shape=(n_samples, pos.shape[1]))
            step = step_size * samples
            return next_rng, reflect(pos, step, self.T, self.b)

        init = gs.tile(self.center[None, :], (n_samples, 1))
        _, samples = jax.lax.fori_loop(0, num_steps, walk, (state, init))
        return samples

    def random_normal_tangent(self, state, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        state, ambiant_noise = bs.random.normal(state=state, size=(n_samples, self.dim))
        chart_noise = self.to_tangent(ambiant_noise, base_point)
        return state, chart_noise

    def random_walk(self, rng, x, t):
        return None
        rng, z = self.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )
        if len(t.shape) == len(x.shape) - 1:
            t = t[..., None]
        tangent_vector = gs.sqrt(t) * z
        samples = self.exp(tangent_vec=tangent_vector, base_point=x)
        return samples

    def belongs(self, x, atol=1e-12):
        return np.all(self.T @ x.T <= self.b[:, None] + atol, axis=0)

    def is_tangent(self, x):
        return True

    def to_tangent(self, vector, base_point):
        inv_metric = self.metric.metric_inverse_matrix(base_point)
        tangent_vector = gs.einsum(
            "...ij,...j->...i",
            gs.linalg.cholesky(inv_metric),
            vector,
        )
        return tangent_vector

    def random_point(self, rng):
        return self.random_uniform(rng)

    @property
    def identity(self):
        return gs.zeros(self.dim)


class ReflectedPolytopeMetric(EuclideanMetric):
    def __init__(self, T, b, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        dim = self.T.shape[1]
        super(ReflectedPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def exp(self, tangent_vec, base_point, **kwargs):
        base_shape = base_point.shape
        base_point = base_point.reshape(-1, base_shape[-1])
        tangent_vec = tangent_vec.reshape(-1, base_shape[-1])
        exp_point = reflect(base_point, tangent_vec, self.T, self.b)
        return exp_point.reshape(
            base_shape
        )  # reflect(base_point, tangent_vec, self.T, self.b)


class HessianPolytopeMetric(RiemannianMetric):
    def __init__(self, T, b, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        dim = self.T.shape[1]
        super(HessianPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def metric_matrix(self, x, eps=1e-4):
        def calc(x):
            res = gs.maximum(self.b - self.T @ x.T, eps)
            return self.T.T @ jax.numpy.diag(res**-2) @ self.T

        return jax.vmap(calc)(x)

    def metric_inverse_matrix_sqrt(self, x):
        return gs.linalg.cholesky(self.metric_inverse_matrix(x))

    def lambda_x(self, x):
        return -1 / 2 * gs.linalg.slogdet(self.metric_matrix(x))[1]

    def grad_logdet_metric_matrix(self, x):
        return jax.grad(self.lambda_x)(x)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Use a retraction instead of the true exponential map."""
        base_point += tangent_vec  # in chart tangent space
        return device_proj(self.T, self.b, base_point)


class HessianCubeMetric(HessianPolytopeMetric):
    def __init__(self, T, b):
        super(HessianCubeMetric, self).__init__(
            T=T, b=b
        )

    def exp(self, tangent_vec, base_point, **kwargs):
        """Use a retraction instead of the true exponential map."""
        base_point += tangent_vec  # in chart tangent space
        return gs.clip(base_point, a_min=-0.1, a_max=0.1)


class HessianTriangleMetric(HessianPolytopeMetric):
    def __init__(self, T, b):
        super(HessianTriangleMetric, self).__init__(
            T=T, b=b
        )
        self.shift = (1 / T.shape[1] * gs.ones(T.shape[1]))
        normal_vec = gs.ones((T.shape[1], 1))
        self.proj = normal_vec @ normal_vec.T / (normal_vec.T @ normal_vec)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Use a retraction instead of the true exponential map."""
        base_point += tangent_vec
        base_point = gs.maximum(base_point, 0)
        mask = self.T[-1, :] @ base_point.T > self.b[-1]
        base_point += mask[:, None] * (self.shift - (base_point @ self.proj.T))
        base_point = gs.clip(base_point, 0, 1)
        return base_point