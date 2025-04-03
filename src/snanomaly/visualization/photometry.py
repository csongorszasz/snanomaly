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
    B = "#1420c7"
    R = "#fc0000"
    I = "#ffaa00"
    g = "#0ceb13"
    r = "#f75e5e"
    i = "#fccc6d"
    g_pr = "#08730b"
    r_pr = "#780101"
    i_pr = "#754e00"

@define
class PlotPhotometry:
    title: str = field(factory=str)
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
            ),
        )

        # set axis labels
        self.figure.update_layout(
            title={"text": self.title, "x": 0.5},
            xaxis={"exponentformat": "none"},
            yaxis={"exponentformat": "power", "showexponent": "all"},
        )
        self.figure.update_xaxes(title_text="MJD", minor={"showgrid": True, "ticks": "inside"})
        self.figure.update_yaxes(title_text="Flux", minor={"showgrid": True, "ticks": "inside"})

    def _get_band_color(self, band: Band):
        try:
            return Color[band.name].value
        except (KeyError, AttributeError):
            return "gray"

    def show(self):
        self.figure.show()

__all__ = ["Bandset", "PlotPhotometry"]
