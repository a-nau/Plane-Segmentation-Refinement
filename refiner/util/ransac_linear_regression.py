import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def ransac_linear_regression(X, y, draw=False):
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression

    try:
        reg = RANSACRegressor(random_state=0).fit(X, y)
    except ValueError:
        try:
            reg = LinearRegression().fit(X, y)
        except:
            return None
    prediction = reg.predict
    if draw:
        plt.xlim([0, 1920])
        plt.ylim([0, 1000])
        plt.scatter(X, y, color='yellowgreen', marker='.', label='Inliers')
        line_y = prediction(X)
        plt.plot(X, line_y, color='navy', label='Linear regressor')
        plt.show()
    if reg.score(X, y) > 0 or np.sum(reg.inlier_mask_) >= 0.8 * reg.inlier_mask_.shape[0]:
        return prediction
    else:
        logger.debug("Regression score was too low ({}) and only {}% inliers, not accepting result."
                    .format(round(reg.score(X, y), 2),
                            100 * round(np.sum(reg.inlier_mask_) / reg.inlier_mask_.shape[0], 2)))
