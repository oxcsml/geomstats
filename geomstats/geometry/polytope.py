"""Euclidean space."""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric


def reflect(rp, s, T, b, eps=1e-5, pass_by_value=True):
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
    # this ends up being pass by reference
    # which may end up with weird behavior
    # so for now I am just copying
    if pass_by_value:
        rp, s = rp.copy(), s.copy()
    # normalize z to a direction and a
    # magnitude; we will use the magnitudes
    # here to continue reflecting until we
    # have traveled the whole distance
    sr = gs.sqrt(gs.sum(s**2, axis=-1))
    s = s / sr[:, None]
    while gs.any(sr > 0):
        # compute the amount we can scale in the
        # direction s before hitting any face,
        # for any of the rp, s vector pairs
        sr_mask = (sr > 0)
        scale = -(((T @ rp.T - b[:, None]) / (T @ s.T)) * sr_mask)
        # we are moving in the "positive" direction
        # of s here, so mask out negative values
        scale_mask = (scale < 0)
        masked_scale = scale_mask * 2 * gs.abs(scale).max() + (1 - scale_mask) * scale
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
        a = gs.clip(sr, 0, a_max - eps)
        rp = rp + a[:, None] * s
        # this is just a test to ensure we are
        # still in the polytope
        assert(gs.all(T @ rp.T <= b[:, None]))
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
        sr = sr - a_max
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

    def __init__(self, constr_path, dataset):
        path = ""
        T = np.loadtxt(f"{constr_path}/T.{dataset}", delimiter=',')
        b = np.loadtxt(f"{constr_path}/b.{dataset}", delimiter=','),
        dim = T.shape[1]
        super(Polytope, self).__init__(
            shape=(dim,), default_point_type="vector", metric=EuclideanMetric(dim)
        )
        self.T, self.b = T, b

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
        return reflect(base_point, tangent_vec, self.T, self.b)
