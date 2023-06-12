"""Euclidean space."""
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import EuclideanMetric, Euclidean
import jax
import numpy as np
import jax.numpy as gs
import geomstats.backend as bs

# from diffrax.misc import bounded_while_loop
from equinox.internal import while_loop

import cvxpy as cp
import jax.experimental.host_callback as hcb

diagm = jax.vmap(gs.diag)


def proj(inp):
    A, b, base_point = inp
    X = cp.Variable(base_point.shape)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(X - base_point)), [A @ X.T <= b[:, None]]
    )
    problem.solve()
    return X.value


def device_proj(A, b, x):
    return hcb.call(
        proj, (A, b, x), result_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
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


def bounded_step(
    base_point, step_dir, step_mag, A, b, eps=1e-6, eps2=1e-8, max_val=1e10
):
    step_size_mask = step_mag > 0
    num, den = A @ base_point.T - b[:, None], A @ step_dir.T
    scale = -stable_div(num, den) * step_size_mask
    scale = gs.clip(scale, -max_val, max_val)
    # we are moving in the "positive" direction
    # of s here, so mask out negative values
    scale_mask = scale <= 0
    masked_scale = scale_mask * max_val + (1 - scale_mask) * scale
    # compute the face we will hit first,
    # e.g. the minimum scaling that lands us
    # on a face
    step_mag_argmax = masked_scale.argmin(axis=0)
    step_mag_max = scale[step_mag_argmax, gs.arange(scale.shape[1])]
    # us either the remaining magnitude sr
    # or the maximum scaling that lands us
    # on a face to scale in the direction s
    # add this to values of rp which we still
    # have magnitude left in their step length
    step_mag = gs.maximum(gs.minimum(step_mag, step_mag_max), 0)
    base_point = base_point + step_mag[:, None] * step_dir
    diff = A @ base_point.T - b[:, None]
    idx = diff >= -eps2
    base_point = base_point + (A.T @ (-(gs.abs(diff) + eps) * idx)).T
    return base_point, step_mag, step_mag_argmax


def reflect(
    base_point, step, A, b, eps=1e-6, eps2=1e-8, max_val=1e10, max_iter=100_000
):
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
        _, _, remaining_step_mag = val
        return gs.any(remaining_step_mag > 0)

    def reflect_body(val):
        # compute the amount we can scale in the
        # direction s before hitting any face,
        # for any of the rp, s vector pairs
        base_point, step_dir, remaining_step_mag = val
        base_point, step_mag, step_mag_argmax = bounded_step(
            base_point,
            step_dir,
            remaining_step_mag,
            A,
            b,
            eps=eps,
            eps2=eps2,
            max_val=max_val,
        )
        # we are going to reflect around the face
        # we land on, so we grab that face from T
        # and normalize it
        normal_face = A[step_mag_argmax, :]
        normal_face = normal_face / gs.sqrt(gs.sum(normal_face**2, axis=-1))[:, None]
        # this is the reflection: note we only
        # need to reflect the direction vector s
        # about the face. for a single vector
        # and a single face we can do that using
        # this eqn: r = s - 2 * dot(s, n) * n
        # where r is the reflection. for the
        # vectorized case we compute the row-wise
        # dot products using gs.sum(s * n, axis=-1)
        step_dir = step_dir - (
            2 * gs.sum(step_dir * normal_face, axis=-1)[:, None] * normal_face
        )
        # because n and s are normalized the
        # resulting s should be normalized too
        # we renormalize for numberical stabilty
        step_dir = step_dir / gs.sqrt(gs.sum(step_dir**2, axis=-1))[:, None]
        # now we subtract the distance we
        # reflected from the magnitude, once this
        # is negative we stop reflecting that
        # vector
        remaining_step_size = remaining_step_mag - step_mag
        return base_point, step_dir, remaining_step_size

    step_mag = gs.sqrt(gs.sum(step**2, axis=-1))
    step_dir = step / step_mag[:, None]
    base_point, _, _ = while_loop(
        reflect_cond,
        reflect_body,
        (base_point, step_dir, step_mag),
        max_steps=max_iter,
        kind="bounded",
    )

    return base_point


class Polytope(Manifold):
    def __init__(
        self,
        T=None,
        b=None,
        npz=None,
        metric=None,
        proj_type=None,
        metric_type="Reflected",
        eps=1e-6,
        **kwargs,
    ):
        if npz is not None:
            data = np.load(npz)
            self.T, self.b = gs.array(data["T"]), gs.array(data["b"])
            if "cube" in npz and proj_type is None:
                proj_type = "cube"
            elif "dirichlet" in npz and proj_type is None:
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
            elif metric_type == "Rejection":
                metric = RejectionPolytopeMetric(self.T, self.b)
            elif metric_type == "Hessian":
                metric = HessianPolytopeMetric(self.T, self.b, eps=eps)
            else:
                raise NotImplementedError()

        super(Polytope, self).__init__(dim=dim, metric=metric)
        self.metric = metric
        # used to compute a point in the interior of the polytope
        # which we can do random walks from to generate random samples
        xc = cp.Variable(self.T.shape[1])
        r = cp.Variable()

        problem = cp.Problem(
            cp.Maximize(r),
            [self.T @ xc.T + r * cp.norm(self.T, axis=1) <= self.b, r >= 0],
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
        return vector

    def random_point(self, rng):
        return self.random_uniform(rng)

    @property
    def identity(self):
        return gs.zeros(self.dim)

    def get_distance_to_boundary(self):
        T, b = self.T, self.b
        vec_T = jax.numpy.sqrt(jax.numpy.sum(T**2, axis=1))

        def distance_to_boundary(x):
            distances = jax.numpy.abs(T @ x.T - b[:, None]) / vec_T[:, None]
            return jax.numpy.min(distances, axis=0)

        return distance_to_boundary

    def distance_to_boundary(self, x):
        T, b = self.T, self.b
        vec_T = jax.numpy.sqrt(jax.numpy.sum(T**2, axis=1))
        distances = jax.numpy.abs(T @ x.T - b[:, None]) / vec_T[:, None]
        return jax.numpy.min(distances, axis=0)


class ReflectedPolytopeMetric(EuclideanMetric):
    def __init__(self, T, b, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        dim = self.T.shape[1]
        super(ReflectedPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def exp(self, tangent_vec, base_point, **kwargs):
        return reflect(base_point, tangent_vec, self.T, self.b)


class RejectionPolytopeMetric(EuclideanMetric):
    def __init__(self, T, b, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        dim = self.T.shape[1]
        super(RejectionPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def exp(self, tangent_vec, base_point, **kwargs):
        new_point = base_point + tangent_vec
        mask = gs.all(self.T @ new_point.T < self.b[:, None], axis=0)
        base_point = (1 - mask[:, None]) * base_point + mask[:, None] * new_point
        return base_point


class HessianPolytopeMetric(RiemannianMetric):
    def __init__(self, T, b, eps=1e-6, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        self.eps = eps
        dim = self.T.shape[1]
        super(HessianPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def metric_matrix(self, x):
        def calc(x):
            res = gs.maximum(self.b - self.T @ x.T, 0) + self.eps
            return self.T.T @ jax.numpy.diag(res**-2) @ self.T

        return jax.vmap(calc)(x)

    def metric_inverse_matrix_sqrt(self, x):
        u, s, v = gs.linalg.svd(self.metric_matrix(x), hermitian=True)
        return u @ diagm(gs.sqrt(s**-1)) @ v

    def lambda_x(self, x):
        # return -1 / 2 * gs.linalg.slogdet(self.metric_matrix(x))[1]
        return 1.0

    def grad_logdet_metric_matrix(self, x):
        return jax.grad(self.lambda_x)(x)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Use a retraction instead of the true exponential map."""
        new_point = base_point + tangent_vec
        mask = gs.all(self.T @ new_point.T < self.b[:, None], axis=0)
        base_point = (1 - mask[:, None]) * base_point + mask[:, None] * new_point
        return base_point

    def norm(self, vector, base_point=None):
        return gs.linalg.norm(vector, axis=-1)

    def squared_norm(self, vector, base_point=None):
        return self.norm(vector, base_point=base_point) ** 2
