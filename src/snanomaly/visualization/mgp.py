from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from attrs import define, field

from snanomaly.models.results.mgp_result import MGPResult
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands
from snanomaly.regression.mgp import MGPInterpolator
from snanomaly.visualization.enums import Color
from snanomaly.visualization.figlayout import FigLayout


@define
class PlotMGP:
    original_bands: Bands = field()
    mgp_result: MGPResult = field()
    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self):
        self.figure.update_layout(FigLayout.light_curves())
        self.set_title(self.mgp_result.sn_name)
        self.set_bands()

    def set_title(self, title: str):
        self.figure.update_layout(title={"text": title, "x": 0.5})

    def set_bands(self, bands: Optional[list[BandEnum]] = None):
        self._clear_figure()
        for band in self.original_bands.get_bands(self.mgp_result.bandset):
            if bands and BandEnum[band.name] not in bands:
                continue
            self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        color = self._get_band_color(band)

        # ground truth
        # TODO: error bars
        self.figure.add_trace(go.Scatter(
            x=band.time,
            y=band.flux,
            mode="markers",
            name=f"Band: {band.name} (observation)",
            marker={"color": color, "symbol": "x"},
        ))
        # interpolation
        band_index = MGPInterpolator.get_band_index(BandEnum(band.name), self.mgp_result.bandset)
        y_mean = self.mgp_result.pred_means[band_index]
        y_std = self.mgp_result.pred_stds[band_index]
        pred_x = MGPInterpolator.get_interval_relative_to_peak(self.mgp_result.peak_time,
                                                               self.mgp_result.days_pre_peak,
                                                               self.mgp_result.days_post_peak)
        self.figure.add_trace(go.Scatter(
            x=pred_x,
            y=y_mean,
            mode="lines",
            name=f"Band: {band.name} (interpolation)",
            line={"color": color},
        ))
        # uncertainty (creating a closed polygon shape for the confidence interval bands)
        self.figure.add_trace(go.Scatter(
            x=np.concatenate([pred_x, pred_x[::-1]]),
            y=np.concatenate([
                y_mean + y_std,
                (y_mean - y_std)[::-1],
            ]),
            fill="toself", # complete path is required
            fillcolor="rgba(0, 0, 0, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
            showlegend=False,
            hoverinfo="skip",
        ))


    def _clear_figure(self):
        # Clear all traces from the figure
        self.figure.data = []

    def _get_band_color(self, band: Band):
        try:
            return Color[band.name].value
        except (KeyError, AttributeError):
            return "gray"

    def show(self, width: int = 600, height: int = 600):
        self.figure.update_layout(
            width=width, height=height,
        )
        self.figure.show()
