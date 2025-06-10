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
    _max_flux_across_all: float = field(default=None, init=False)

    def __attrs_post_init__(self):
        for b in self.preprocessed_bands:
            if b.is_normalized:
                b.denormalize()

        self.figure.update_layout(FigLayout.light_curves())
        pred_min, pred_max = self.prediction_interval.min(), self.prediction_interval.max()
        self.figure.update_xaxes(range=[pred_min, pred_max], tickmode="array",
                                 tickvals=np.arange(pred_min, pred_max + 1, (pred_max - pred_min) / 6), tickformat="d")
        self.figure.update_yaxes(range=[0 - self.max_flux_across_all / 40, 1.5 * self.max_flux_across_all])
        self.figure.update_layout(margin=dict(l=0, r=0, t=60, b=100))
        self.set_title(self.int_result.sn_name)
        self.set_bands()

    @property
    def max_flux_across_all(self):
        if self._max_flux_across_all:
            return self._max_flux_across_all
        og_max = max(
            band.flux.max() for band in self.original_bands.get_bands(self.int_result.bandset)
        )
        preprocessed_max = max(
            band.flux.max() for band in self.preprocessed_bands
        )
        self._max_flux_across_all = max(og_max, preprocessed_max)
        return self._max_flux_across_all

    @property
    def prediction_interval(self):
        return BaseInterpolator.get_interval_relative_to_peak(self.int_result.peak_time,
                                                              self.int_result.days_pre_peak,
                                                              self.int_result.days_post_peak)

    def set_title(self, main_title: str, subtitle: str = ""):
        self.figure.update_layout(title={
            "text": main_title,
            "x": 0.1,
            "font": {"size": 20},
        })
        self.figure.update_layout(title_subtitle={
            "text": subtitle,
            "font": {"color": "gray", "size": 10},
        })

    def set_subtitle(self, text: str):
        pass

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
                    name="prediction",
                    line={"color": color},
                    legendgrouptitle={
                        "text": band.display_name,
                        "font": {"size": 12},
                    },
                    legendgroup=band.display_name,
                    legendrank=10,
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
                        "size": 5,
                        "line": {"width": 1, "color": color},
                    },
                    x=band.ignored_upperlimits_time,
                    y=band.ignored_upperlimits_flux,
                    name="ignored upper limit",
                    hoverinfo="text",
                    legendgroup=band.display_name,
                    legendrank=40,
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
                        "line": {"width": 0.5, "color": "white"},
                    },
                    x=band.time[band.upperlimit].ravel(),
                    y=band.flux[band.upperlimit],
                    name="converted upper limit",
                    hoverinfo="text",
                    error_y={
                        "type": "data",
                        "array": band.e_flux,
                        "visible": True,
                        "color": color,  # Match error bar color with marker color
                        "thickness": 1,
                    },
                    legendgroup=band.display_name,
                    legendrank=30,
                ),
            )
        # ground truth
        if gnd_truth:
            self.figure.add_trace(
                go.Scatter(
                    x=band.time[~band.upperlimit].ravel(),
                    y=band.flux[~band.upperlimit],
                    mode="markers",
                    name="ground truth",
                    marker={"color": color, "symbol": "x", "size": 5, "line": {"width": 0.5, "color": "white"}},
                    error_y={
                        "type": "data",
                        "array": band.e_flux,
                        "visible": True,
                        "color": color,  # Match error bar color with marker color
                        "thickness": 1,
                    },
                    legendgroup=band.display_name,
                    legendrank=20,
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

    @staticmethod
    def show_grid(
        plotters: list[PlotInterpolation],
        nrows: int,
        ncols: int,
        width_per_subfig: int,
        height_per_subfig: int,
        output_path: str = "grid_plot.png",
    ):
        import io

        from PIL import Image

        # Convert each figure to image bytes
        imgs = []
        for p in plotters:
            img_bytes = p.figure.to_image(format="png", width=width_per_subfig, height=height_per_subfig)
            imgs.append(img_bytes)

        # Define the grid dimensions for the final image
        grid_width = ncols * width_per_subfig
        grid_height = nrows * height_per_subfig

        # Create a new blank image (RGBA to handle potential transparency, white background)
        grid_image = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 255))

        # Convert each figure to image bytes and paste it onto the grid
        for idx, p in enumerate(plotters):
            if idx >= nrows * ncols:
                break  # Don't process more images than grid cells

            img_bytes = p.figure.to_image(format="png", width=width_per_subfig, height=height_per_subfig)
            img = Image.open(io.BytesIO(img_bytes))

            # Calculate position in grid
            row = idx // ncols
            col = idx % ncols
            x = col * width_per_subfig
            y = row * height_per_subfig

            # Paste the image onto the grid
            grid_image.paste(img, (x, y))

        # Save the final grid image
        # If the output path is for a format that doesn't support alpha (like JPEG), convert to RGB
        if output_path.lower().endswith((".jpg", ".jpeg")):
            grid_image = grid_image.convert("RGB")
        grid_image.save(output_path)
