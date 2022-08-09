"""Product of Riemannian metrics.

Define the metric of a product manifold endowed with a product metric.
"""

import joblib

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.geometry.riemannian_metric import RiemannianMetric


import jax
from functools import partial
from jax import jit
from jax import numpy as jnp

import numpy as np

class ProductSameRiemannianMetric(RiemannianMetric):
    """Jax based vmapped product metircs over the same manifold"""

    def __init__(self, metric, mul, default_point_type="vector", **kwargs):
        self.metric = metric
        self.mul = mul
        self.dim = self.mul * self.metric.dim

        super().__init__(
            dim=self.dim,
            signature=(metric.signature[0] * self.mul, metric.signature[1] * self.mul),
            default_point_type=default_point_type,
        )

    def _iterate_over_metrics(self, func, kwargs, in_axes=-2, out_axes=-2):
        method = getattr(self.metric, func)
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

    def exp(self, tangent_vec, base_point):
        return self._iterate_over_metrics(
            "exp",
            {
                "tangent_vec": tangent_vec,
                "base_point": base_point,
            },
        )

    def log(self, point, base_point=None, point_type=None):
        return self._iterate_over_metrics(
            "log",
            {
                "point": point,
                "base_point": base_point,
                # "point_type": point_type,
            },
        )

    def squared_norm(self, vector, base_point=None):
        return gs.sum(
            self._iterate_over_metrics(
                "squared_norm",
                {
                    "vector": vector,
                    "base_point": base_point,
                },
                out_axes=-1,
            ),
            axis=-1,
        )


class ProductRiemannianMetric(RiemannianMetric):
    """Jax based vmapped product metircs over the same manifold"""

    def __init__(self, metrics, default_point_type="vector", **kwargs):
        self.metrics = metrics
        self.metric_dims = [
            metric.embedding_space.dim if hasattr(metric, "embedding_space") 
            else metric.dim for metric in self.metrics
        ]
        self.cum_dims = np.cumsum(
            [metric.embedding_space.dim if hasattr(metric, "embedding_space") 
             else metric.dim for metric in self.metrics]
        ).tolist()
        self.dim = sum(self.metric_dims)

        super().__init__(
            dim=self.dim,
            signature=(
                sum([metric.signature[0] for metric in self.metrics]), 
                sum([metric.signature[1] for metric in self.metrics])
            ),
            default_point_type=default_point_type,
        )

    def _iterate_over_metrics(self, func, kwargs, in_axes=-2, out_axes=-2):
        out = []
        for i, metric in enumerate(self.metrics):
            metric_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, "reshape"):
                    metric_kwargs[key] = value[..., (self.cum_dims[i] - self.metric_dims[i]):self.cum_dims[i]]
                else:
                    metric_kwargs[key] = value
            out.append(getattr(metric, func)(**metric_kwargs))
        return gs.concatenate(out, axis=-1)

    def exp(self, tangent_vec, base_point):
        return self._iterate_over_metrics(
            "exp",
            {
                "tangent_vec": tangent_vec,
                "base_point": base_point,
            },
        )

    def log(self, point, base_point=None, point_type=None):
        return self._iterate_over_metrics(
            "log",
            {
                "point": point,
                "base_point": base_point,
            },
        )

    def squared_norm(self, vector, base_point=None):
        return gs.sum(
            self._iterate_over_metrics(
                "squared_norm",
                {
                    "vector": vector,
                    "base_point": base_point,
                },
                out_axes=-1,
            ),
            axis=-1,
        )
    
class ProductSameRiemannianMetric2(ProductRiemannianMetric):
    """Jax based vmapped product metircs over the same manifold"""

    def __init__(self, metric, mul, default_point_type="vector", **kwargs):
        self.metric = metric
        self.mul = mul

        super().__init__(
            metrics=self.mul * [self.metric],
            default_point_type=default_point_type,
        )

    def _iterate_over_metrics(self, func, kwargs, in_axes=-2, out_axes=-2):
        method = getattr(self.metric, func)
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

    
        
        
        
