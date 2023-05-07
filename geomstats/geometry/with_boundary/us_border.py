import os
import pickle
import pandas as pd
import jax.numpy as jnp
from jax import jit
from functools import partial

import geoviews as gv
import geoviews.feature as gf

from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric


"""
Define conversion functions between spherical and cartesian coordinates
lng = phi, lat = theta in https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
"""

@jit
def spherical2cartesian(lng, lat):
    """
    Converts the coordinates of a point on the unit sphere from spherical to Cartesian. 
    """

    x = jnp.sin(lat) * jnp.cos(lng)
    y = jnp.sin(lat) * jnp.sin(lng)
    z = jnp.cos(lat)
    
    # assert np.allclose(np.sqrt(np.square(x) + np.square(y) + np.square(z)), 1), "Not all transformed points on unit sphere."
    
    return x, y, z

@jit
def cartesian2spherical(x, y, z):
    """
    Converts the coordinates of a point on the unit sphere from Cartesian to spherical.  
    """

    # assert np.allclose(np.sqrt(np.square(x) + np.square(y) + np.square(z)), 1), "Not all points on unit sphere."
    
    lng = jnp.arctan2(y, x)
    lat = jnp.arctan2(jnp.hypot(x,y), z)
    
    # map all anggles to within [0, 2pi) radians
    lng %= 2 * jnp.pi
    lat %= 2 * jnp.pi

    return lng, lat

def get_national_boundary_fn(pkl="/data/ziz/not-backed-up/fishman/score-sde/continental_us.pkl"):
    """
    Precompute a bunch of polytope-specific stuff.
    """

    with open(pkl, "rb") as f:
        us = pickle.load(f)

    us = us.simplify(0)

    # get polygon and reference point in polygon (in this case the centroid is
    # inside the polygon so we're just using that, doesn't not hold in general)

    polygon = jnp.array(us.boundary.coords)
    centroid = jnp.array(us.centroid.coords[0])

    # convert latitudes and longitudes to radians and 
    # map longitudes from [-pi, pi] to [0, 2pi]
    # and latitudes from [-pi/2°, pi/2°] to [0, pi]

    long_lat_transform = lambda x: jnp.deg2rad(x) + jnp.array([jnp.pi, jnp.pi/2])

    polygon = long_lat_transform(polygon)
    centroid = long_lat_transform(centroid)

    # test coordinate conversion functions
    assert jnp.allclose(
        jnp.stack(cartesian2spherical(*spherical2cartesian(*polygon.T)), 1),
        polygon
    ), "Error in conversion functions."

    # also convert them from spherical to Cartesian coordinates
    polygon_c = jnp.stack(spherical2cartesian(*polygon.T), 1)
    centroid_c = jnp.array(spherical2cartesian(*centroid))

    (polygon.shape, centroid.shape), (polygon_c.shape, centroid_c.shape)

    # to efficiently check whether a query point is in the polytope
    # we change their bases to align with the reference point inside the polytope
    z_axis = centroid_c
    y_axis = jnp.cross(z_axis, polygon_c[0])
    x_axis = jnp.cross(y_axis, z_axis)

    z_axis /= jnp.linalg.norm(z_axis)
    y_axis /= jnp.linalg.norm(y_axis)
    x_axis /= jnp.linalg.norm(x_axis)

    Q = jnp.vstack([x_axis, y_axis, z_axis])

    # map polygon, centroid and query points into the new reference frame
    centroid_c = Q @ centroid_c
    polygon_c = polygon_c @ Q.T

    # and also precompute the new spherical coordinates
    polygon_qf = jnp.stack(cartesian2spherical(*polygon_c.T), 1)

    # get edge longitudes in query frame and precompute the 
    # sorted edge longitudes for each subsequent vertex pair
    edge_lngs = jnp.stack([polygon_qf[:, 0], jnp.roll(polygon_qf[:, 0], shift=-1, axis=0)], 1)
    edge_lngs = jnp.sort(edge_lngs, axis=1)
    edge_lngs_diff = jnp.expand_dims((edge_lngs[:, 1] - edge_lngs[:, 0]) >= jnp.pi, 0)

    # precompute the poles corresponding to each vertex edge
    # and the dot product of these poles with the centroid
    poles_c = jnp.cross(polygon_c, jnp.roll(polygon_c, shift=-1, axis=0))
    poles_precomp = jnp.dot(poles_c, centroid_c)


    @partial(jit, static_argnames=("poles_c", "poles_precomp", "edge_lngs", "edge_lngs_diff"))
    def is_in_boundary(
        q_c,
        poles_c=poles_c, 
        poles_precomp=poles_precomp,
        edge_lngs=edge_lngs,
        edge_lngs_diff=edge_lngs_diff
    ):
        """
        Checks if an Nx3 array of spherical points in extrinsic coordinates are within the boundary.
        """

        # convert query points to Cartesian coordinates and change reference frame
        # q_c = jnp.stack(spherical2cartesian(*q.T), 1)
        q_c = q_c @ Q.T
        
        # also generate spherical coordinates of the new reference frame
        q_qf = jnp.stack(cartesian2spherical(*q_c.T), 1)
        
        # the first condition checks whether the longitude of a query point in the 
        # reference frame is bounded by the longitudes of an edge's vertices
        lower_bound = jnp.expand_dims(q_qf[:, 0], 1) >= edge_lngs[:, 0]
        upper_bound = jnp.expand_dims(q_qf[:, 0], 1) <= edge_lngs[:, 1]

        strike_con = jnp.logical_and(lower_bound, upper_bound)
        strike_con = jnp.logical_xor(strike_con, edge_lngs_diff)
        
        # the second contion checks whether the query and reference point are on different sides 
        # of the edge by using it as an equator to divide the sphere into two hemispheres
        # strict inequality counts points on boundary as not crossing it
        hemi_con = ((q_c @ poles_c.T) * poles_precomp) < 0
        
        # using these conditions we get the number of polygon edges the geodesic between 
        # a query point and a reference point inside the polygon crosses. 
        # If this number is even, the query point is inside the polygon.
        is_inside = jnp.logical_not(jnp.mod(jnp.logical_and(strike_con, hemi_con).sum(1), 2))
        
        return is_inside
    
    return is_in_boundary

class BoundedHypersphere(Hypersphere):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim):
        super(BoundedHypersphere, self).__init__(dim)
        is_in_boundary = get_national_boundary_fn()
        self.metric = RejectionHypersphereMetric(dim, is_in_boundary)


class RejectionHypersphereMetric(HypersphereMetric):
    """Class for the metric of the n-dimensional hypersphere with boundary.
    
    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim, is_in_boundary):
        super().__init__(dim)
        self.is_in_boundary = is_in_boundary


    def exp(self, tangent_vec, base_point, **kwargs):
        new_basepoint = super().exp(tangent_vec, base_point, **kwargs)
        mask = self.is_in_boundary(new_basepoint)[:, None]
        return (1 - mask) * base_point + mask * new_basepoint

