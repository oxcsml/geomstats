from timeit import timeit
from functools import partial

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
import jax
import numpy as np

SPHERE2 = Hypersphere(dim=2)
METRIC = SPHERE2.metric


def plot_and_save_video(
    trajectories, pdf=None, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    if pdf:
        sphere.plot_heatmap(ax, pdf)
    points = gs.to_ndarray(trajectories[0], to_ndim=2)
    sphere.draw(ax, color=color, marker=".")
    scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
    with writer.saving(fig, out, dpi=dpi):
        for points in trajectories[1:]:
            points = gs.to_ndarray(points, to_ndim=2)
            scatter.remove()
            scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
            writer.grab_frame()


def vMF_pdf(x, mu, kappa):
    """https://gist.github.com/marmakoide/6f55ff99f14c896399c460a38f72c99a"""
    constant = kappa / ((2 * np.pi) * (1. - np.exp(-2. * kappa)))
    return constant * np.exp(kappa * (np.dot(mu, x) - 1.))


# def brownian_motion(previous_x, delta, N):
#     for i in range(N):
#         ambiant_noise = gs.random.normal(size=(previous_x.shape[0], SPHERE2.dim + 1))
#         noise = delta * SPHERE2.to_tangent(vector=ambiant_noise, base_point=previous_x)
#         x = METRIC.exp(tangent_vec=noise, base_point=previous_x)
#     return x


@jax.jit
def brownian_motion(previous_x, delta, N):
    rng = jax.random.PRNGKey(0)

    def body(step, val):
        rng, x = val
        rng, step_rng = jax.random.split(rng)
        ambiant_noise = jax.random.normal(step_rng, (x.shape[0], SPHERE2.dim + 1))
        noise = delta * SPHERE2.to_tangent(vector=ambiant_noise, base_point=x)
        x = METRIC.exp(tangent_vec=noise, base_point=x)
        return rng, x

    _, x = jax.lax.fori_loop(0, N, body, (rng, previous_x))
    return x


def main():
    """Run gradient descent on a sphere."""
    # gs.random.seed(1985)

    N = 100
    n_samples = 1000
    delta = 0.1
    mu = gs.array([1, 0, 0])
    kappa = 15
    initial_point = SPHERE2.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
    previous_x = initial_point

    x = brownian_motion(previous_x, delta, N)

    # plot_and_save_video(trajectories, pdf=partial(vMF_pdf, mu=mu, kappa=kappa), out="forward.mp4")

    

if __name__ == "__main__":
    main()
    time = timeit(lambda : main(), number=10)
    print(time) 