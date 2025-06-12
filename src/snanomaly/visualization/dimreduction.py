from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from attrs import define, field


@define
class PlotDimreduction:
    data_2d: np.ndarray = field()
    labels: Optional[np.ndarray] = field(default=None)
    ids: Optional[list[str]] = field(default=None)
    title_text: Optional[str] = field(default="")
    xlabel: str = field(default="x")
    ylabel: str = field(default="y")
    label_name: str = field(default="Event")
    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self) -> None:
        num_points = self.data_2d.shape[0]
        self.figure.update_layout(
            title={
                "text": self.title_text,
                "x": 0.5,
                "y": 0.93,
                "font": {"size": 20},
            },
            xaxis_title={
                "text": self.xlabel,
                "font": {"size": 16},
            },
            yaxis_title={
                "text": self.ylabel,
                "font": {"size": 16},
            },
            template="plotly_white",
            xaxis=dict(
                showline=True,
                showgrid=True,
                linecolor="rgb(128, 128, 128)",
                linewidth=1,
                mirror=True,
                ticks="inside",
                tickfont={"size": 14},
                gridcolor="rgb(220,220,220)",
            ),
            yaxis=dict(
                showline=True,
                showgrid=True,
                linecolor="rgb(128, 128, 128)",
                linewidth=1,
                mirror=True,
                ticks="inside",
                tickfont={"size": 14},
                gridcolor="rgb(220,220,220)",
            ),
            legend={
                "font": {"size": 14},
                "title_font": {"size": 15},
            },
            margin=go.layout.Margin(l=50, r=50, b=50, t=80, pad=4),
            annotations=[
                go.layout.Annotation(
                    text=f"Number of points: {num_points}",
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    bordercolor="black",
                    borderwidth=1,
                    font={"size": 10, "color": "black"},
                ),
            ],
        )
        self._plot_data()

    def set_title(self, text: str) -> None:
        self.title_text = text
        self.figure.update_layout(title={"text": self.title_text, "x": 0.5})

    def set_xaxis_label(self, text: str) -> None:
        self.xlabel = text
        self.figure.update_xaxes(title_text=self.xlabel)

    def set_yaxis_label(self, text: str) -> None:
        self.ylabel = text
        self.figure.update_yaxes(title_text=self.ylabel)

    def _plot_data(self) -> None:
        self.figure.data = []  # Clear existing traces

        # Prepare hover texts
        point_hover_texts: Optional[list[str]] = None
        base_hover_info = [f"x: {x_val:.2f}<br>y: {y_val:.2f}" for x_val, y_val in zip(self.data_2d[:, 0], self.data_2d[:, 1], strict=False)]

        if self.ids is not None:
            if self.labels is not None:
                point_hover_texts = [
                    f"ID: {id_}<br>{self.label_name}: {lbl}<br>{base}"
                    for id_, lbl, base in zip(self.ids, self.labels, base_hover_info, strict=False)
                ]
            else:
                point_hover_texts = [f"ID: {id_}<br>{base}" for id_, base in zip(self.ids, base_hover_info, strict=False)]
        elif self.labels is not None:
            point_hover_texts = [
                f"{self.label_name}: {lbl}<br>{base}" for lbl, base in zip(self.labels, base_hover_info, strict=False)
            ]
        else:
            point_hover_texts = base_hover_info # Default hover text with coordinates only

        hoverinfo = "text" if point_hover_texts is not None else "skip" # Use "skip" if no text, rely on default x,y,name

        self.figure.add_trace(
            go.Scatter(
                x=self.data_2d[:, 0],
                y=self.data_2d[:, 1],
                mode="markers",
                hovertext=point_hover_texts,
                hoverinfo=hoverinfo if point_hover_texts else "x+y",
                marker={"size": 6, "opacity": 0.7, "line": {"width": 0.5, "color": "DarkSlateGrey"}},
                name="Data points",
            ),
        )

    def show(self, width: int = 600, height: int = 600) -> None:
        self.figure.update_layout(
            width=width,
            height=height,
        )
        self.figure.show()

    def write_image(self, path: pathlib.Path, width: int = 600, height: int = 600):
        self.figure.write_image(path, width=width, height=height)
