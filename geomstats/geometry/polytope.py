"""Euclidean space."""
from geomstats.geometry.euclidean import Euclidean
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

def reflect(r, sn, T, b, eps=1e-6, eps2=1e-10, max_val=1e10, pass_by_value=True, max_iter=100_000):
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


def to_params(x):
    # this is jittable
    diag = x[1:, :] - x[0, :]
    l = gs.sqrt(gs.sum((x[1:-1] - x[2:])**2, axis=1))
    d = gs.sqrt(gs.sum((x[0]-x[-1])**2))
    r = gs.sqrt(gs.sum((diag)**2, axis=1))
    diag /= r[:, None]

    n = gs.cross(diag[:-1, :], diag[1:, :])
    n /= gs.sqrt(gs.sum(n**2, axis=1))[:, None]

    tau = []
    for j in range(n.shape[0] - 1):
        theta = gs.sign(n[j + 1] @ diag[j]) * gs.arccos(n[j + 1] @ n[j])
        tau.append(theta)
    tau = gs.array(tau)
    return r, tau, l, d

def to_euclidean(r, tau, l, x0, xn, xn1):
    # not jittable
    m = r.shape[0] + 1

    diag = gs.ones((m - 1, 3))
    n = gs.ones((m - 2, 3))
    gamma = gs.ones(m-3)

    diag[m-2] = xn - x0
    diag[m-3] = xn1 - x0
    diag /= gs.sqrt(gs.sum((diag)**2, axis=1))[:, None]

    n[m-3] = gs.cross(diag[-1], diag[-2])
    n /= gs.sqrt(gs.sum((n)**2, axis=1))[:, None]
    
    x = gs.zeros((m, 3))
    x[0], x[m-2], x[m-1] = x0, xn1, xn

    for j in range(m-2-2, -1, -1):
        n[j] = diag[j + 1] * (diag[j + 1] @ n[j+1]) + \
               gs.cos(-tau[j]) * gs.cross(gs.cross(diag[j + 1], n[j+1]), diag[j + 1]) + \
               gs.sin(-tau[j]) * gs.cross(diag[j + 1], n[j+1])
        n /= gs.sqrt(gs.sum((n)**2, axis=1))[:, None]
        gamma[j] = gs.arccos((r[j]**2 + r[j+1]**2 - l[j]**2) / (2 * r[j] * r[j+1]))
        diag[j] = n[j] * (n[j] @ diag[j+1]) + \
                  gs.cos(gamma[j]) * gs.cross(gs.cross(n[j], diag[j+1]), n[j]) + \
                  gs.sin(gamma[j]) * gs.cross(n[j], diag[j+1])
        diag /= gs.sqrt(gs.sum((diag)**2, axis=1))[:, None]
        x[j+1] = x0 + r[j] * diag[j]
    return x

def get_constraints(l, D):
    m = l.shape[0] + 2
    T = np.zeros((3 * m - 8, m - 3))
    b = np.zeros((3 * m - 8))
    T[0, 0], b[0] = 1, l[0] + l[1]
    T[1, 0], b[1] = -1, -gs.abs(l[0] - l[1])
    i = 2
    k = 1
    for j in range(0, m-4):
        T[i, j], T[i, j + 1], b[i] = 1, -1, l[j + k]
        T[i + 1, j], T[i + 1, j + 1], b[i + 1] = -1, 1, l[j + k]
        T[i + 2, j], T[i + 2, j + 1], b[i + 2] = -1, -1, -l[j + k]
        i += 3
    T[-2, -1], b[-2] = 1, l[-3] + D
    T[-1, -1], b[-1] = -1, - gs.abs(l[-3] - D)
    return T, b



class Polytope(Euclidean):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, n_links=16, npz=None):
        if npz is not None:
            data = np.load(npz)
        else:
            data = np.load(f"/data/ziz/not-backed-up/fishman/score-sde/data/walk.0.{n_links}.npz")
        self.T, self.b = gs.array(data['T']), gs.array(data['b'])
        dim = self.T.shape[1]
        super(Polytope, self).__init__(dim=dim)
        c = np.zeros((self.T.shape[1],))
        res = linprog(
            c, 
            A_ub=self.T, b_ub=self.b[:, None], 
            bounds=(None, None)
        )
        self.center = res.x

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential, which is simply the addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n]
            Point from which the exponential is computed.

        Returns
        -------
        point : array-like, shape=[..., n]
            Group exponential.
        """
        # if not self.belongs(tangent_vec):
            # raise ValueError("The update must be of the same dimension")
        base_shape = base_point.shape
        base_point = base_point.reshape(-1, base_shape[-1])
        tangent_vec = tangent_vec.reshape(-1, base_shape[-1])
        exp_point = reflect(base_point, tangent_vec, self.T, self.b)
        return exp_point.reshape(base_shape)
    
    @property
    def log_volume(self):
        return self.metric.log_volume
    
    def random_uniform(self, state, n_samples=1, step_size=10., num_steps=100_000):
        def walk(i, carry):
            rng, pos = carry
            rng, next_rng = jax.random.split(rng)  
            samples = jax.random.normal(rng, shape=(n_samples, pos.shape[1]))
            step = step_size * samples
            return next_rng, reflect(pos, step, self.T, self.b) 
        
        init = gs.tile(self.center[None, :], (n_samples, 1))
        _, samples = jax.lax.fori_loop(
            0, num_steps, walk, (state, init)
        )
        return samples
        
    def belongs(self, x, atol=1e-12):
        return self.T @ x.T <= self.b[:, None] + atol
        
        
