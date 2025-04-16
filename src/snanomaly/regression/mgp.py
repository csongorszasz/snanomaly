from __future__ import annotations

import numpy as np
from attrs import define, field
from multistate_kernel import MultiStateKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from snanomaly.models.sncandidate.bands import Bands


@define
class MGPRegressor:
    """Multi-output Gaussian Process."""

    bands: Bands = field()
    regressor: GaussianProcessRegressor = field(init=False)

    def __attrs_post_init__(self):
        self._construct_regressor()
        self._fit_regressor()

    def _construct_regressor(self):
        # kernels
        k1 = RBF(length_scale=1.0, length_scale_bounds=(0.01, 10))
        k2 = RBF(length_scale=1.0, length_scale_bounds=(0.01, 10))
        k3 = RBF(length_scale=1.0, length_scale_bounds=(0.01, 10))

        # lower triangular scale matrix
        scale = np.array([[1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
        scale_bounds = [np.full_like(scale, -2.0), np.full_like(scale, 2.0)]

        # multi state kernel
        ms_kernel = MultiStateKernel(kernels=[k1, k2, k3], scale=scale, scale_bounds=scale_bounds)

        # alpha

        # optimizer

        # regressor
        self.regressor = GaussianProcessRegressor(
            kernel=ms_kernel,
            # alpha=...,
            # optimizer=...,
            n_restarts_optimizer=0,
            normalize_y=True,
            # random_state=...
        )

    def _fit_regressor(self):
        # prepped_bands = []
        # for band in sn.photometry.bands.get_bands(Bandset.gri):
        #     # 1-day binning
        #     binned = band.binned(bin_width=1)
        #     # normalize flux vectors by maximum values
        #     binned.flux = binned.flux / np.max(binned.flux)
        #     prepped_bands.append(binned)
        # X = np.concatenate(
        #     [
        #         np.vstack((np.zeros(prepped_bands[0].nr_observations), prepped_bands[0].time)).T,
        #         np.vstack((np.ones(prepped_bands[1].nr_observations), prepped_bands[1].time)).T,
        #         np.vstack((2 * np.ones(prepped_bands[2].nr_observations), prepped_bands[2].time)).T,
        #     ]
        # )
        # y = np.concatenate([band.flux for band in prepped_bands])
        # self.regressor.fit(X, y)
        pass

    def predict(self, time_lower_bound: int, time_upper_bound: int):
        # time_new = np.linspace(np.min(prepped_bands[0].time), np.max(prepped_bands[0].time), 100)
        # X_new_all = np.concatenate([np.vstack((i * np.ones(len(time_new)), time_new)).T for i in range(3)])
        # y_pred_all, y_std_all = gp.predict(X_new_all, return_std=True)
        # y_pred_all = y_pred_all.reshape(3, len(time_new)).T
        # y_std_all = y_std_all.reshape(3, len(time_new)).T
        pass
