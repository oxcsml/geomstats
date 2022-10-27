"""Euclidean space."""
from geomstats.geometry.euclidean import EuclideanMetric, Euclidean
import jax
import numpy as np
import jax.numpy as gs
from scipy.optimize import linprog

from diffrax.misc import bounded_while_loop

# todo: if b is always either M, 1 or M, N then
# you dont need to expand dims and you can handle
# different b with the same T matrix: this is nice
# because in our setting the T matrix is always the
# same; all that changes is the b values! 


def stable_div(num, den, eps=1e-10):
    return gs.sign(num) * gs.sign(den) * gs.exp(gs.log(gs.abs(num) + eps) - gs.log(gs.abs(den) + eps))


def reflect(r, sn, T, b, eps=1e-6, eps2=1e-10, max_val=1e10, max_iter=100_000):
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
        sr_mask = (sr > 0)
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
        diff = (T @ rp.T - b[:, None])
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
    rp, s, sr = bounded_while_loop(
        reflect_cond,
        reflect_body,
        (r, sn, sr),
        max_iter
    )
    
    return rp


class Polytope(Euclidean):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, T=None, b=None, npz=None, metric=None, metric_type="Reflected", **kwargs):
        if npz is not None:
            data = np.load(npz)
            self.T, self.b = gs.array(data['T']), gs.array(data['b'])
        elif T is not None and b is not None:
            self.T, self.b = gs.array(T), gs.array(b)
        else:
            raise ValueError("You need either the inequality matrices or "
                             "an archive pointing to them")
        dim = self.T.shape[1]
        if metric is None:
            if metric_type == "Reflected":
                metric = ReflectedPolytopeMetric(self.T, self.b)
            elif metric_type == "Hessian":
                metric = HessianPolytopeMetric(self.T, self.b)
            else:
                raise NotImplementedError

        super(Polytope, self).__init__(dim=dim, metric=metric)
        self.metric = metric
        # used to compute a point in the interior of the polytope
        # which we can do random walks from to generate random samples
        c = np.zeros((self.T.shape[1],))
        res = linprog(
            c, 
            A_ub=self.T, b_ub=self.b[:, None], 
            bounds=(None, None)
        )
        self.center = res.x

    def exp(self, tangent_vec, base_point=None):
        return self.metric.exp(tangent_vec, base_point)
    
    @property
    def log_volume(self):
        return self.metric.log_volume
    
    def random_uniform(self, state, n_samples=1, step_size=1., num_steps=10_000):
        def walk(_, carry):
            rng, pos = carry
            rng, next_rng = jax.random.split(rng)  
            samples = jax.random.normal(rng, shape=(n_samples, pos.shape[1]))
            step = step_size * samples
            return next_rng, reflect(step, pos, self.T, self.b)
        
        init = gs.tile(self.center[None, :], (n_samples, 1))
        _, samples = jax.lax.fori_loop(
            0, num_steps, walk, (state, init)
        )
        return samples

    def random_walk(self, rng, x, t):
        rng, z = self.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )
        if len(t.shape) == len(x.shape) - 1:
            t = t[..., None]
        tangent_vector = gs.sqrt(t) * z
        samples = self.exp(tangent_vec=tangent_vector, base_point=x)
        return samples
        
    def belongs(self, x, atol=1e-12):
        return self.T @ x.T <= self.b[:, None] + atol
        
        
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
        return exp_point.reshape(base_shape) #reflect(base_point, tangent_vec, self.T, self.b)


class HessianPolytopeMetric(EuclideanMetric):
    def __init__(self, T, b, default_point_type="vector", **kwargs):
        self.T, self.b = T, b
        dim = self.T.shape[1]
        super(HessianPolytopeMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def metric_matrix(self, x, t, z, eps=1e-6):
        def calc(x):
            res = gs.maximum(self.b - self.T @ x.T, eps)
            return self.T.T @ gs.diag(res**-2) @ self.T
        return jax.vmap(calc)(x)

    def metric_inverse_matrix(self, x, t, z, eps=1e-6):
        def calc(x):
            res = gs.maximum(self.b - self.T @ x.T, eps)
            return gs.linalg.inv(self.T.T @ gs.diag(res**-2) @ self.T)
        return jax.vmap(calc)(x)

    def metric_inverse_matrix_sqrt(self, x, t, z, eps=1e-6):
        def calc(x):
            res = gs.maximum(self.b - self.T @ x.T, eps)
            return gs.linalg.cholesky(gs.linalg.inv(self.T.T @ gs.diag(res**-2) @ self.T))
        return jax.vmap(calc)(x)

    def exp(self, tangent_vec, base_point, eps=1e-8, **kwargs):
        base_point += tangent_vec
        diff = (self.T @ base_point.T - self.b[:, None])
        idx = diff >= 0
        return base_point + (self.T.T @ (-(gs.abs(diff) + eps) * idx)).T
