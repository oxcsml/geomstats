import jax.numpy as gs


def to_params(x):
    """
    x is a set of euclidean coordinates of chains with
    fixed end points
    """
    diag = x[1:, :] - x[0, :]
    l = gs.sqrt(gs.sum((x[1:-1] - x[2:]) ** 2, axis=1))
    d = gs.sqrt(gs.sum((x[0] - x[-1]) ** 2))
    r = gs.sqrt(gs.sum((diag) ** 2, axis=1))
    diag /= r[:, None]

    n = gs.cross(diag[:-1, :], diag[1:, :])
    n /= gs.sqrt(gs.sum(n ** 2, axis=1))[:, None]

    tau = []
    for j in range(n.shape[0] - 1):
        theta = gs.sign(n[j + 1] @ diag[j]) * gs.arccos(n[j + 1] @ n[j])
        tau.append(theta)
    tau = gs.array(tau)
    return r, tau, l, d


def to_euclidean(r, tau, l, x0, xn1, xn):
    """
    r is a point in the polytope
    tau is a point in the torus of the same dim as the polytope
    s is a point on the unit sphere (R^3)
    l is the set of lengths of the lings
    d is the distance from the start to end points
    x0 is the start point
    xn is the endpoint
    """
    m = r.shape[0] + 2

    diag = gs.ones((m - 1, 3))
    n = gs.ones((m - 2, 3))
    gamma = gs.ones(m - 3)

    diag = diag.at[m - 2].set(xn - x0)
    diag = diag.at[m - 3].set(xn1 - x0)
    diag = diag.at[m - 2].set(xn - x0)
    diag /= gs.sqrt(gs.sum((diag) ** 2, axis=1))[:, None]

    n = n.at[m - 3].set(gs.cross(diag[-2], diag[-1]))
    n /= gs.sqrt(gs.sum((n) ** 2, axis=1))[:, None]

    x = gs.zeros((m, 3))
    x = x.at[0].set(x0)
    x = x.at[m - 2].set(xn1)
    x = x.at[m - 1].set(xn)

    for j in range(m - 2 - 2, -1, -1):
        n = n.at[j].set(
            diag[j + 1] * (diag[j + 1] @ n[j + 1]) + \
            gs.cos(-tau[j]) * gs.cross(gs.cross(diag[j + 1], n[j + 1]), diag[j + 1]) + \
            gs.sin(-tau[j]) * gs.cross(diag[j + 1], n[j + 1])
        )
        n /= gs.sqrt(gs.sum((n) ** 2, axis=1))[:, None]
        gamma = gamma.at[j].set(
            gs.arccos((r[j] ** 2 + r[j + 1] ** 2 - l[j] ** 2) / (2 * r[j] * r[j + 1]))
        )
        diag = diag.at[j].set(
            n[j] * (n[j] @ diag[j + 1]) + \
            gs.cos(-gamma[j]) * gs.cross(gs.cross(n[j], diag[j + 1]), n[j]) + \
            gs.sin(-gamma[j]) * gs.cross(n[j], diag[j + 1])
        )
        diag /= gs.sqrt(gs.sum((diag) ** 2, axis=1))[:, None]
        x = x.at[j + 1].set(x0 + r[j] * diag[j])
    return x


def get_constraints(l, D):
    m = l.shape[0] + 2
    T = gs.zeros((3 * m - 8, m - 3))
    b = gs.zeros((3 * m - 8))
    T[0, 0], b[0] = 1, l[0] + l[1]
    T[1, 0], b[1] = -1, -gs.abs(l[0] - l[1])
    i = 2
    k = 1
    for j in range(0, m - 4):
        T[i, j], T[i, j + 1], b[i] = 1, -1, l[j + k]
        T[i + 1, j], T[i + 1, j + 1], b[i + 1] = -1, 1, l[j + k]
        T[i + 2, j], T[i + 2, j + 1], b[i + 2] = -1, -1, -l[j + k]
        i += 3
    T[-2, -1], b[-2] = 1, l[-2] + D
    T[-1, -1], b[-1] = -1, -gs.abs(l[-2] - D)
    return T, b