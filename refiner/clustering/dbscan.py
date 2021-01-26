import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def dbscan_with_masked_image(image, eps=0.5, min_samples=3):
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale!")
    masked_points = np.where(image > 0)
    if len(masked_points) > 0:
        X = np.column_stack(tuple((masked_points[1], masked_points[0])))
        if X.shape[0] > 0:
            class_dict = dbscan_with_points(X, eps, min_samples)
            return class_dict
        else:
            return dict()
    else:
        return dict()


def dbscan_with_points(X, eps=0.5, min_samples=3):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    class_dict = {}
    k_max = 0
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        if np.sum(labels == k) > np.sum(labels == k_max):
            k_max = k
        xy = scaler.inverse_transform(X[class_member_mask & core_samples_mask])
        class_dict[k] = xy

    class_dict['max'] = class_dict[k_max]
    class_dict.pop(k_max)
    return class_dict
