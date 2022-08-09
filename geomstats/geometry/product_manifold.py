"""Product of manifolds."""

import joblib

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric
from geomstats.geometry.hypersphere import Hypersphere

import numpy as np

import jax
import jax.numpy as jnp
from geomstats.geometry.product_riemannian_metric import ProductSameRiemannianMetric
from jax.scipy.linalg import block_diag

from functools import partial
from jax import jit

from collections import namedtuple

EmbeddingSpace = namedtuple("EmbeddingSpace", "dim")

@partial(jit, static_argnums=[0, 1, 2])
def pull(i, cum_dims, manifold_dims, value):
    return jax.lax.slice_in_dim(
        value, cum_dims[i] - manifold_dims[i], self.cum_dims[i], axis=-1
    )

class ProductSameManifold(Manifold):
    def __init__(self, manifold, mul, metric=None, default_point_type="vector", **kwargs):
        self.manifold = manifold
        self.mul = mul
        self.dim = self.mul * self.manifold.dim

        if metric is None:
            metric = ProductSameRiemannianMetric(self.manifold.metric, self.mul)
        self.metric = metric

        super(ProductSameManifold, self).__init__(
            dim=self.dim, metric=metric, default_point_type=default_point_type, **kwargs
        )
        self.dim = self.mul * self.manifold.dim
        if isinstance(manifold, EmbeddedManifold):
            self.embedding_space = ProductSameManifold(manifold.embedding_space, mul)

    def _iterate_over_manifolds(self, func, kwargs, in_axes=-2, out_axes=-2):
        method = getattr(self.manifold, func)
        in_axes = []
        args_list = []
        for key, value in kwargs.items():
            if hasattr(value, "reshape"):
                # value : shape=[..., mul*dim] -> shape=[..., mul, dim]
                value = value.reshape((*value.shape[:-1], self.mul, -1))
                in_axes.append(-2)
            else:
                in_axes.append(None)
            args_list.append(value)
        # out = jax.vmap(method, in_axes=in_axes, out_axes=out_axes)(**args)
        # NOTE: Annoyingly cannot pass named arguments with in_axes! https://github.com/google/jax/issues/7465
        out = jax.vmap(method, in_axes=in_axes, out_axes=out_axes)(*args_list)
        out = out.reshape((*out.shape[:out_axes], -1))
        # value : shape=[..., mul, dim] -> shape=[..., mul*dim]
        return out

    def belongs(self, point, atol=gs.atol):
        belongs = self._iterate_over_manifolds(
            "belongs",
            {"point": point, "atol": atol},
            out_axes=-1,
        )
        belongs = gs.all(belongs, axis=-1)
        return belongs

    def projection(self, point):
        projected_point = self._iterate_over_manifolds(
            "projection",
            {"point": point},
        )
        return projected_point

    def to_tangent(self, vector, base_point):
        tangent_vec = self._iterate_over_manifolds(
            "to_tangent",
            {"vector": vector, "base_point": base_point},
        )
        return tangent_vec

    def random_uniform(self, state, n_samples=1):
        if self.mul > 1:
            new_state = jax.random.split(state, num=self.mul)
            state = new_state.reshape((-1))
        samples = self._iterate_over_manifolds(
            "random_uniform",
            {"state": state, "n_samples": n_samples},
        )
        return samples

    def random_point(self, n_samples=1, bound=1.0):
        samples = self._iterate_over_manifolds(
            "random_point",
            {"n_samples": n_samples, "bound": bound},
        )
        return samples

    def random_walk(self, rng, x, t):
        if self.mul > 1:
            new_state = jax.random.split(rng, num=self.mul)
            rng = new_state.reshape((-1))
        if isinstance(x, jax.numpy.ndarray):
            t = t[:, None] * jnp.ones(self.mul)[None, :]
        walks = self._iterate_over_manifolds("random_walk", {"rng": rng, "x": x, "t": t})
        return walks

    def _log_heat_kernel(self, x0, x, t, n_max):
        t = t * jnp.ones((*x0.shape[:-1], self.mul))

        log_probs = self._iterate_over_manifolds(
            "_log_heat_kernel", {"x0": x0, "x": x, "t": t, "n_max": n_max}, out_axes=-1
        )
        return jnp.sum(log_probs, axis=-1)

    def random_normal_tangent(self, state, base_point, n_samples=1):
        size = (n_samples, self.mul * self.manifold.embedding_space.dim)
        state, ambiant_noise = gs.random.normal(state=state, size=size)
        return state, self.to_tangent(vector=ambiant_noise, base_point=base_point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        is_tangent = self._iterate_over_manifolds(
            "is_tangent",
            {"vector": vector, "base_point": base_point, "atol": atol},
            out_axes=-1,
        )
        is_tangent = gs.all(is_tangent, axis=-1)
        return is_tangent

    def exp(self, tangent_vec, base_point, **kwargs):
        return self._iterate_over_manifolds(
            "exp",
            {"tangent_vec": tangent_vec, "base_point": base_point},
        )

    def log(self, point, base_point, **kwargs):
        return self._iterate_over_manifolds(
            "log",
            {"point": point, "base_point": base_point},
        )

    def div_free_generators(self, x):
        generators = self._iterate_over_manifolds("div_free_generators", {"x": x})
        # return generators
        def block_diag_generators(generators):
            gens = jnp.split(generators, self.mul, axis=1)
            return block_diag(*gens)

        block_diag_generators = jax.vmap(block_diag_generators, in_axes=0, out_axes=0)
        return block_diag_generators(generators)

    @property
    def log_volume(self):
        # TODO: Double check
        return self.mul * self.manifold.log_volume

    # TODO: I CBA to write product groups rn, this is a hacky af way to get the isom_group.dim thing to work for moser flows

    @property
    def isom_group(self):
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        return dotdict(dim=self.manifold.isom_group.dim * self.mul)

    
class ProductManifold(Manifold):
    def __init__(self, dim=None, manifolds=None, metric=None, default_point_type="vector", **kwargs):
        self.manifolds = manifolds
        print(manifolds)
        self.dim = sum([manifold.dim for manifold in self.manifolds]) if dim is None else dim

        if metric is None:
            metric = ProductRiemannianMetric([manifold.metric for manifold in self.manifolds])
        self.metric = metric

        super(ProductManifold, self).__init__(
            dim=self.dim, metric=metric, default_point_type=default_point_type, **kwargs
        )
        self.manifold_dims = [manifold.dim for manifold in self.manifolds]
        self.cum_dims = np.cumsum(self.manifold_dims).tolist()

    # @partial(jit, static_argnums=[0, 1])
    def _iterate_over_manifolds(self, func, kwargs, in_axes=-2, out_axes=-2):
        out = []
        for i, manifold in enumerate(self.manifolds):
            manifold_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, "reshape"):
                    if value.shape[-1] == len(self.manifolds):
                        manifold_kwargs[key] = value[..., i]
                    else:
                        manifold_kwargs[key] = value[..., (self.cum_dims[i] - self.manifold_dims[i]):self.cum_dims[i]]
                else:
                    manifold_kwargs[key] = values
                print(manifold_kwargs[key])
            out.append(getattr(manifold, func)(**manifold_kwargs))
        if None in out:
            return None
        return gs.concatenate(out, axis=-1)

    def belongs(self, point, atol=gs.atol):
        belongs = self._iterate_over_manifolds(
            "belongs",
            {"point": point, "atol": atol},
            out_axes=-1,
        )
        belongs = gs.all(belongs, axis=-1)
        return belongs

    def projection(self, point):
        projected_point = self._iterate_over_manifolds(
            "projection",
            {"point": point},
        )
        return projected_point

    def to_tangent(self, vector, base_point):
        tangent_vec = self._iterate_over_manifolds(
            "to_tangent",
            {"vector": vector, "base_point": base_point},
        )
        return tangent_vec

    def random_uniform(self, state, n_samples=1):
        if len(self.manifolds) > 1:
            new_state = jax.random.split(state, num=len(self.manifolds))
            state = new_state.reshape((-1))
        samples = self._iterate_over_manifolds(
            "random_uniform",
            {"state": state, "n_samples": n_samples},
        )
        return samples

    def random_point(self, n_samples=1, bound=1.0):
        samples = self._iterate_over_manifolds(
            "random_point",
            {"n_samples": n_samples, "bound": bound},
        )
        return samples

    def random_walk(self, rng, x, t):
        if len(self.manifolds) > 1:
            new_state = jax.random.split(rng, num=len(self.manifolds))
            rng = new_state# .reshape((-1))
        if isinstance(x, jax.numpy.ndarray):
            t = t[:, None] * jnp.ones(len(self.manifolds))[None, :]
        walks = self._iterate_over_manifolds("random_walk", {"rng": rng, "x": x, "t": t})
        return walks

    def _log_heat_kernel(self, x0, x, t, n_max):
        t = t * jnp.ones((*x0.shape[:-1], len(self.manifolds)))

        log_probs = self._iterate_over_manifolds(
            "_log_heat_kernel", {"x0": x0, "x": x, "t": t, "n_max": n_max}, out_axes=-1
        )
        return jnp.sum(log_probs, axis=-1)

    def random_normal_tangent(self, state, base_point, n_samples=1):
        size = (n_samples, sum(self.manifold_dims))
        state, ambiant_noise = gs.random.normal(state=state, size=size)
        return state, self.to_tangent(vector=ambiant_noise, base_point=base_point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        is_tangent = self._iterate_over_manifolds(
            "is_tangent",
            {"vector": vector, "base_point": base_point, "atol": atol},
            out_axes=-1,
        )
        is_tangent = gs.all(is_tangent, axis=-1)
        return is_tangent

    def exp(self, tangent_vec, base_point, **kwargs):
        return self._iterate_over_manifolds(
            "exp",
            {"tangent_vec": tangent_vec, "base_point": base_point},
        )

    def log(self, point, base_point, **kwargs):
        return self._iterate_over_manifolds(
            "log",
            {"point": point, "base_point": base_point},
        )

    def div_free_generators(self, x):
        generators = self._iterate_over_manifolds("div_free_generators", {"x": x})
        # return generators
        def block_diag_generators(generators):
            gens = jnp.split(generators, len(self.manifolds), axis=1)
            return block_diag(*gens)

        block_diag_generators = jax.vmap(block_diag_generators, in_axes=0, out_axes=0)
        return block_diag_generators(generators)

    @property
    def log_volume(self):
        # TODO: Double check
        return sum([manifold.log_volume for manifold in self.manifolds])

    # TODO: I CBA to write product groups rn, this is a hacky af way to get the isom_group.dim thing to work for moser flows

    @property
    def isom_group(self):
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        return dotdict(dim=sum([manifold.isom_group.dim for manifold in self.manifolds]))

class EmbeddedProductManifold(ProductManifold, EmbeddedManifold):
    def __init__(self, manifolds, metric=None, default_point_type="vector", **kwargs):
        self.manifolds = manifolds
        manifold_dims = [manifold.embedding_space.dim if hasattr(manifold, "embedding_space")
                              else manifold.dim for manifold in self.manifolds]
        dim = sum(manifold_dims)
        embedding_space = ProductManifold(
            dim=dim, # wrong dim but itll be fixed
            manifolds = [
                manifold.embedding_space if hasattr(manifold, "embedding_space") 
                else manifold for manifold in self.manifolds        
            ]
        )
        super(EmbeddedProductManifold, self).__init__(
            dim=dim,
            manifolds=manifolds,
            embedding_space=embedding_space,
            **kwargs
        )
        self.manifold_dims = manifold_dims
        self.dim = dim
        print("TASA", dim)
    
    
class Product2Manifolds(EmbeddedProductManifold):
    def __init__(
        self, manifold_a, manifold_b, **kwargs
    ):
        super(Product2Manifolds, self).__init__(
            manifolds=[manifold_a, manifold_b],
            **kwargs
        )

    