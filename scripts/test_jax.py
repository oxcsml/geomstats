from timeit import timeit

import jax
import geomstats.backend as gs
from geomstats.geometry.connection import Connection
from geomstats.geometry.hypersphere import Hypersphere

connection = Connection(2)
hypersphere = Hypersphere(2)
connection.christoffels = hypersphere.metric.christoffels

base_point = gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]])
point = gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]])

max_shape = point.shape if point.ndim == 3 else base_point.shape
n_steps = 5#0
step = 'rk4'

@jax.jit
def objective(velocity):
    """Define the objective function."""
    velocity = gs.array(velocity)
    velocity = gs.cast(velocity, dtype=base_point.dtype)
    velocity = gs.reshape(velocity, max_shape)
    delta = connection.exp(velocity, base_point, n_steps, step) - point
    return gs.sum(delta ** 2)

obj = gs.autodiff.value_and_grad(objective)

obj(point - base_point)
time = timeit(lambda : obj(point - base_point), number=10)
print(time)