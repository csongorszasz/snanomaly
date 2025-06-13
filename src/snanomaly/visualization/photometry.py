from __future__ import annotations

from typing import Optional

from attrs import define, field
from plotly import graph_objects as go

from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum
from snanomaly.models.sncandidate.photometry import Photometry
from snanomaly.visualization.enums import Color
from snanomaly.visualization.figlayout import FigLayout


@define
class PlotPhotometry:
    photometry: Photometry = field()
    title: str = field(factory=str)
    figure: go.Figure = field(factory=go.Figure)

    def __attrs_post_init__(self):
        self.figure.update_layout(FigLayout.light_curves())

        self.set_bands()
        self.set_title(self.title)

    def set_title(self, title: str):
        self.figure.update_layout(title={"text": title, "x": 0.5})

    def set_bands(self, band_names: Optional[list[BandEnum]] = None):
        self._clear_figure()
        if band_names:
            for band_name in band_names:
                band = self.photometry.bands.get_band(band_name)
                if band:
                    self._add_band_to_figure(band)
        else:
            for band in self.photometry.bands.get_bands():
                if band.nr_observations > 0:
                    self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        color = self._get_band_color(band)

        # plot observation data points
        self.figure.add_trace(
            go.Scatter(
                mode="markers",
                marker={
                    "symbol": "circle",
                    "color": color,
                    "size": 5,
                    "line": {"width": 0.25, "color": "black"},
                },
                x=band.time[~band.upperlimit],
                y=band.flux[~band.upperlimit],
                error_y={
                    "type": "data",
                    "array": band.e_flux,
                    "visible": True,
                    "color": color,  # Match error bar color with marker color
                    "thickness": 1.5,
                },
                name=f"{band.name.replace('_pr', "'")} (observation)",
                hoverinfo="text",
            ),
        )

        # plot upper limits
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
                name=f"{band.name.replace("_pr", "'")} (upper limit)",
                hoverinfo="text",
            ),
        )

    def _clear_figure(self):
        # Clear all traces from the figure
        self.figure.data = []

    def _get_band_color(self, band: Band):
        return Color[band.name].value

    def show(self, width: int = 600, height: int = 600):
        self.figure.update_layout(
            width=width, height=height,
        )
        self.figure.show(renderer="browser")

__all__ = ["PlotPhotometry"]
