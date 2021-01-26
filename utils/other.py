import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_n_unique_rows(edge):
    new = [tuple(row) for row in edge]
    edge_unique = np.unique(new)
    edge_size = edge_unique.shape[0]
    return edge_size


def get_unique_rows(edge):
    edge_unique = np.unique(edge, axis=0)
    return edge_unique


def get_depth_of_nested_list(l):
    depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1
    return depth(l)
