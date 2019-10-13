import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import voronoi_plot_2d

DEFAULT_GEO_ARGS = {
    'lng_bound': (121.44, 121.64),
    'lat_bound': (24.95, 25.15),
    'lng_granularity': 0.02,
    'lat_granularity': 0.02,
}
DEFAULT_PLOT_ARGS = {
    'alpha': 0.1,
    's': 10,
    'linewidth': 0
}

DEFAULT_VOR_ARGS = {
    'show_points': True,
    'show_vertices': False,
    'line_colors': 'orange'
}

def scatter_by_fixed_area(x, y,
                          lng_bound=DEFAULT_GEO_ARGS['lng_bound'],
                          lat_bound=DEFAULT_GEO_ARGS['lat_bound'],
                          lng_granularity=DEFAULT_GEO_ARGS['lng_granularity'],
                          lat_granularity=DEFAULT_GEO_ARGS['lat_granularity']):
    fig = plt.figure(figsize=(12, 12), dpi=100)
    plt.scatter(x, y, **DEFAULT_PLOT_ARGS)

    ax = fig.add_subplot(111)

    # set the lng and lat as ticks
    lng_ticks = np.arange(lng_bound[0], lng_bound[1], lng_granularity)
    lat_ticks = np.arange(lat_bound[0], lat_bound[1], lat_granularity)
    ax.xaxis.set_ticks(lng_ticks)
    ax.yaxis.set_ticks(lat_ticks)
    ax.grid(True)
    return ax

def plot_cluster_centroid(x, y, centroid,
                          centroid_marker='*',
                          title=f'Clusters found by HDBSCAN',
                          title_font_size=24):
    # label y = -1 means outlier, use colour (0, 0, 0)
    palette = sns.color_palette('bright', np.unique(y).max() + 2)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in y]

    fig = plt.figure(figsize=(12, 12), dpi=100)
    plt.scatter(x.T[1], x.T[0], c=colors, **DEFAULT_PLOT_ARGS)
    plt.scatter(centroid.T[1], centroid.T[0], c='green', marker='*')

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title(title, fontsize=title_font_size)
    return fig

def plot_voronoi_plot_2d(vor, **kwargs):
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(111)

    show_points = kwargs.get('show_points', DEFAULT_VOR_ARGS['show_points'])
    show_vertices = kwargs.get('show_vertices', DEFAULT_VOR_ARGS['show_vertices'])
    line_colors = kwargs.get('line_colors', DEFAULT_VOR_ARGS['line_colors'])
    return voronoi_plot_2d(vor,
                           ax=ax,
                           show_points=show_points,
                           show_vertices=show_vertices,
                           line_colors=line_colors)

def plot_voronoi_with_bound(polygons, points, polygon_bound, **kwargs):
    """plot multiple shapely polygon 

    Warning:
        polygon should be a generator
    """
    fig = plt.figure(figsize=(12, 12), dpi=100)

    # draw polygon bound
    bound_x, bound_y = zip(*polygon_bound.exterior.coords)
    plt.plot(bound_x, bound_y, 'b-')

    # draw polygons
    _pts_x = [p.x for p in points]
    _pts_y = [p.y for p in points]
    plt.plot(_pts_x, _pts_y, 'b.')
    for p in polygons:
        if p.geom_type != 'Polygon':
            print(p)
            continue
        x, y = zip(*p.exterior.coords)
        plt.plot(x, y, 'r-')

    return fig
