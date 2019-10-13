
import numpy as np
import geopandas as gpd

import hdbscan
from shapely.geometry import Polygon, Point
from collections import defaultdict
from itertools import compress
from sklearn.neighbors.nearest_centroid import NearestCentroid


DEFAULT_HDBSCAN_ARGS={
    'REF_LAG': 25.05,
    'REF_LNG': 121.54,
    'SHIFT_MUL_SCALE': 100,
    'MIN_CLUSTER_SIZE': 3
}

def check_points_in_polygon(x, y, polygon):
    assert len(x) == len(y)
    return [polygon.contains(Point(*coor)) for coor in zip(x, y)]

def get_hdbscan_cluster_label(lat_lng_coor,
                              ref_lat=DEFAULT_HDBSCAN_ARGS['REF_LAG'],
                              ref_lng=DEFAULT_HDBSCAN_ARGS['REF_LNG'],
                              **kwargs):
    """Get the hdbscan clustering label

    Transform the actual coordinate to shift coordinate then apply hdbscan
    """
    # get the shift coor
    shift_coor = lat_lng_coor.copy()
    shift_coor[:, 0] = (shift_coor[:, 0] - ref_lat)*100
    shift_coor[:, 1] = (shift_coor[:, 1] - ref_lng)*100

    # apply clustering algorithm
    min_cluster_size = kwargs.get('min_cluster_size', DEFAULT_HDBSCAN_ARGS['MIN_CLUSTER_SIZE'])
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = cluster_model.fit_predict(shift_coor)
    return cluster_labels

def get_centroid(lat_lng_coor, labels):
    model = NearestCentroid()
    model.fit(lat_lng_coor, labels)
    return model.centroids_

def get_voronoi_polygon(vor, diameter):
    """Generate Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the input points.

    The polygons for the infinite regions are large enough that 
    all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    ref: https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region
    """
    centroid = vor.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), idx_rv in zip(vor.ridge_points, vor.ridge_vertices):
        u, v = sorted(idx_rv)
        if u == -1:
            # infinite ridge starting ar ridge point with index v
            # equidistant from input points with indexes p and q.
            t = vor.points[q] - vor.points[p] # tangent vector
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal vector
            midpoint = vor.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)
    for i, r in enumerate(vor.point_region):
        region = vor.regions[r]
        # finite polygon
        if -1 not in region:
            yield Polygon(vor.vertices[region])
            continue
        # infinite polygon
        inf = region.index(-1) # index of vertex at infinity
        j = region[(inf - 1) % len(region)] # index of previous vertex
        k = region[(inf + 1) % len(region)] # index of next vertex
        if j == k:
            # region has one voronoi vertex with two ridges
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # region has two voronoi vertex, each with one ridge
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # length of ridges needs for the extra edge to lie at least
        # 'diamter' away from all voronoi vertices
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # polygon consists of finite part + extra edge
        finite_part = vor.vertices[region[inf+1:] + region[:inf]]
        extra_edge = [vor.vertices[j] + dir_j * length,
                      vor.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def map_points_in_polygon(polygons, points):
    """
    FIXME: if polygon contains multiple points

    Return GeoDataFrame
    """
    df_polygon = gpd.GeoDataFrame({'geometry_polygon': polygons}, geometry='geometry_polygon')
    df_points = gpd.GeoDataFrame({'geometry_point': points}, geometry='geometry_point')
    df_geo = gpd.tools.sjoin(df_polygon, df_points, how='left', op='intersects')
    assert df_polygon.shape[0] == df_points.shape[0] == df_geo.shape[0]
    df_geo['index_right'] = df_geo['index_right'].apply(lambda x: df_points.geometry_point.iloc[x])
    df_geo.columns = ['geometry_polygon', 'geometry_point']
    return df_geo

def get_bounded_polygon(polygons, points, bounded_polygon):
    """Intersect with bounded polygon may leads to multipolygon

    consider the voronoi points as the valid polygon case
    """
    for polygon, point in zip(polygons, points):
        assert polygon.contains(point), 'Make sure all the points are in the corresponding polygon'
        bounded_subpolygon = polygon.intersection(bounded_polygon)
        if bounded_subpolygon.geom_type == 'MultiPolygon':
            idx_contains_point = [poly.contains(point) for poly in bounded_subpolygon]
            bounded_subpolygon = list(compress(bounded_subpolygon, idx_contains_point))[0]
        yield bounded_subpolygon, point
