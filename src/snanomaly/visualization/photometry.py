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
    figure: go.Figure = field(factory=go.Figure)

    def __attrs_post_init__(self):
        # set axis labels
        self.figure.update_layout(
            xaxis={"exponentformat": "none"},
            yaxis={"exponentformat": "power", "showexponent": "all"},
        )
        self.figure.update_xaxes(title_text="MJD (Modified Julian Date)", minor={"showgrid": True, "ticks": "inside"})
        self.figure.update_yaxes(title_text=r"$Flux (\text{erg}\,\text{s}^{-1}\,\text{Hz}^{-1}\,\text{cm}^{-1})$", minor={"showgrid": True, "ticks": "inside"})
        # self.figure.update_yaxes(title_text="Flux (erg s^(-1) Hz^(-1) cm^(-1))", minor={"showgrid": True, "ticks": "inside"})

    def set_title(self, title: str):
        self.figure.update_layout(title={"text": title, "x": 0.5})

    def set_bands(self, bandsets: list[Bandset]):
        self._clear_figure()
        for bandset in bandsets:
            for band in bandset.band_references:
                self._add_band_to_figure(band)

    def _add_band_to_figure(self, band: Band):
        # color = self._get_band_color(band)

        # Create a scatter plot for the band data
        self.figure.add_trace(
            go.Scatter(
                mode="markers",
                marker={"symbol": "circle", "line_width": 1, "size": 5},
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

__all__ = ["Bandset", "PlotPhotometry"]
