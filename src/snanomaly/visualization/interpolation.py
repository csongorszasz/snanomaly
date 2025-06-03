from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
from attrs import define, field

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.models.results.interpolation_result import InterpolationResult
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands
from snanomaly.visualization.enums import Color
from snanomaly.visualization.figlayout import FigLayout


@define
class PlotInterpolation:
    original_bands: Bands = field()
    int_result: InterpolationResult = field()
    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self):
        self.figure.update_layout(FigLayout.light_curves())
        self.figure.update_xaxes(range=[self.prediction_interval.min(), self.prediction_interval.max()])
        self.set_title(self.int_result.sn_name)
        self.set_bands()

    @property
    def prediction_interval(self):
        return BaseInterpolator.get_interval_relative_to_peak(self.int_result.peak_time,
                                                              self.int_result.days_pre_peak,
                                                              self.int_result.days_post_peak)

    def set_title(self, title: str):
        self.figure.update_layout(title={"text": title, "x": 0.5})

    def set_bands(self, bands: Optional[list[BandEnum]] = None):
        self._clear_figure()
        for band in self.original_bands.get_bands(self.int_result.bandset):
            if bands and BandEnum[band.name] not in bands:
                continue
            self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        color = self._get_band_color(band)

        # ground truth
        self.figure.add_trace(
            go.Scatter(
                x=band.time[~band.upperlimit],
                y=band.flux[~band.upperlimit],
                mode="markers",
                name=f"Band: {band.name} (observation)",
                marker={"color": color, "symbol": "x"},
                error_y={
                    "type": "data",
                    "array": band.e_flux,
                    "visible": True,
                    "color": color,  # Match error bar color with marker color
                    "thickness": 1.5,
                },
            ),
        )
        # upper limits
        self.figure.add_trace(
            go.Scatter(
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "color": color,
                    "size": 5,
                    "line": {"width": 0.25, "color": "black"},
                },
                x=band.time[band.upperlimit],
                y=band.flux[band.upperlimit],
                name=f"{band.name} (upper limit)",
                hoverinfo="text",
            ),
        )
        # interpolation
        band_index = BaseInterpolator.get_band_index(BandEnum(band.name), self.int_result.bandset)
        y = self.int_result.preds[band_index]
        pred_x = self.prediction_interval
        self.figure.add_trace(go.Scatter(
            x=pred_x,
            y=y,
            mode="lines",
            name=f"Band: {band.name} (interpolation)",
            line={"color": color},
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
