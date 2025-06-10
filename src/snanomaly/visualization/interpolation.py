from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from attrs import define, field

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.models.results.interpolation_result import InterpolationResult
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands
from snanomaly.visualization.enums import Color
from snanomaly.visualization.figlayout import FigLayout
from snanomaly.visualization.util import color_to_rgba


@define
class PlotInterpolation:
    original_bands: Bands = field()
    preprocessed_bands: list[Band] = field()
    int_result: InterpolationResult = field()
    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self):
        for b in self.preprocessed_bands:
            b.denormalize()

        self.figure.update_layout(FigLayout.light_curves())
        self.figure.update_xaxes(range=[self.prediction_interval.min(), self.prediction_interval.max()])
        self.figure.update_yaxes(minallowed=0)
        self.set_title(self.int_result.sn_name)
        self.set_bands()

    @property
    def prediction_interval(self):
        return BaseInterpolator.get_interval_relative_to_peak(self.int_result.peak_time,
                                                              self.int_result.days_pre_peak,
                                                              self.int_result.days_post_peak)

    def set_title(self, text: str):
        self.figure.update_layout(title={"text": text, "x": 0.5})

    def set_subtitle(self, text: str):
        self.figure.update_layout(title_subtitle={"text": text, "font": {"color": "gray", "size": 10}})

    def set_bands(self, bands: Optional[list[BandEnum]] = None):
        self._clear_figure()
        for band in self.preprocessed_bands:
            if bands and BandEnum[band.name] not in bands:
                continue
            self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        color = self._get_band_color(band)
        original_band = self.original_bands.get_band(band.name)

        gnd_truth = True
        upper_limits = True
        interpolation_result = True
        stds = True

        # interpolation
        if interpolation_result:
            band_index = BaseInterpolator.get_band_index(BandEnum(band.name), self.int_result.bandset)
            y = self.int_result.preds[band_index]
            pred_x = self.prediction_interval.ravel()
            self.figure.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=y,
                    mode="lines",
                    name=f"{band.name.replace('_pr', "'")} (prediction)",
                    line={"color": color},
                ),
            )

            # uncertainty (creating a closed polygon shape for the confidence interval bands)
            if stds and self.int_result.stds:
                band_index = BaseInterpolator.get_band_index(BandEnum(band.name), self.int_result.bandset)
                y = self.int_result.preds[band_index]
                std = self.int_result.stds[band_index]
                self.figure.add_trace(
                    go.Scatter(
                        x=np.concatenate([pred_x, pred_x[::-1]]),
                        y=np.concatenate(
                            [
                                y + std,
                                (y - std)[::-1],
                            ],
                        ),
                        fill="toself",  # complete path is required
                        fillcolor=color_to_rgba(color, 0.2),
                        line={"color": "rgba(255, 255, 255, 0)"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )
        if upper_limits:
            # ignored upper limits
            self.figure.add_trace(
                go.Scatter(
                    mode="markers",
                    marker={
                        "symbol": "triangle-down",
                        "color": "black",
                        "size": 7,
                        "line": {"width": 1.5, "color": color},
                    },
                    x=band.ignored_upperlimits_time,
                    y=band.ignored_upperlimits_flux,
                    name=f"{original_band.name.replace("_pr", "'")} (ignored upper limit)",
                    hoverinfo="text",
                ),
            )
            # kept upper limits
            self.figure.add_trace(
                go.Scatter(
                    mode="markers",
                    marker={
                        "symbol": "triangle-down",
                        "color": color,
                        "size": 5,
                    },
                    x=band.time[band.upperlimit].ravel(),
                    y=band.flux[band.upperlimit],
                    name=f"{band.name.replace("_pr", "'")} (converted upper limit)",
                    hoverinfo="text",
                    error_y={
                        "type": "data",
                        "array": band.e_flux,
                        "visible": True,
                        "color": color,  # Match error bar color with marker color
                        "thickness": 1,
                    },
                ),
            )
        # ground truth
        if gnd_truth:
            self.figure.add_trace(
                go.Scatter(
                    x=original_band.time[~original_band.upperlimit],
                    y=original_band.flux[~original_band.upperlimit],
                    mode="markers",
                    name=f"{original_band.name.replace('_pr', "'")} (ground truth)",
                    marker={"color": color, "symbol": "x", "size": 5},
                    error_y={
                        "type": "data",
                        "array": original_band.e_flux,
                        "visible": True,
                        "color": color,  # Match error bar color with marker color
                        "thickness": 1,
                    },
                ),
            )

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
