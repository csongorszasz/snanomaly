from __future__ import annotations

from enum import Enum

from attrs import define, field
from plotly import graph_objects as go

from snanomaly.models.sncandidate.band import Band


@define
class Bandset:
    """Group `Band` instance references to create a bandset."""

    band_references: list[Band] = field(factory=list)


class Color(Enum):
    B = "blue"
    R = "red"
    I = "pink"
    g = "green"
    r = "brown"
    i = "yellow"
    g_pr = "purple"
    r_pr = "orange"
    i_pr = "cyan"

@define
class PlotPhotometry:
    figure: go.Figure = field(factory=go.Figure)

    def add_bands(self, bandsets: list[Bandset]):
        for bandset in bandsets:
            for band in bandset.band_references:
                self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        color = self._get_band_color(band)

        # Create a scatter plot for the band data
        self.figure.add_trace(
            go.Scatter(
                x=band.time,
                y=band.flux,
                error_y={
                    "type": "data",
                    "array": band.e_flux,
                    "visible": True,
                },
                name=band.name,
                marker={"color": color},
            ),
        )

    def _get_band_color(self, band: Band):
        try:
            return Color[band.name].value
        except (KeyError, AttributeError):
            return "gray"

    def show(self):
        self.figure.show()

__all__ = ["Bandset", "PlotPhotometry"]
