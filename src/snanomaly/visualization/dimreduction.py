from __future__ import annotations

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
    label_name: str = field(default="Label")
    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self) -> None:
        self.figure.update_layout(
            title={"text": self.title_text, "x": 0.5},
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            template="plotly_white",
            xaxis=dict(
                showline=True,
                showgrid=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=1,
                ticks="outside",
                gridcolor="rgb(230,230,230)",
            ),
            yaxis=dict(
                showline=True,
                showgrid=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=1,
                ticks="outside",
                tickfont={"family": "Times New Roman", "size": 12, "color": "rgb(82, 82, 82)"},
                gridcolor="rgb(230,230,230)",
            ),
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

        if self.labels is None:
            self.figure.add_trace(
                go.Scatter(
                    x=self.data_2d[:, 0],
                    y=self.data_2d[:, 1],
                    mode="markers",
                    hovertext=point_hover_texts,
                    hoverinfo=hoverinfo if point_hover_texts else "x+y",
                    marker={"size": 5},
                    name="Data points",
                ),
            )
        else:
            unique_labels = np.unique(self.labels)
            is_categorical = False
            if self.labels.dtype == object or self.labels.dtype.kind in "SU":  # strings
                is_categorical = True
            elif self.labels.dtype.kind in "iu":  # integers
                if len(unique_labels) <= 20 and len(unique_labels) < len(self.labels) / 2 : # Heuristic for categorical integers
                    is_categorical = True
            # Float is continuous by default

            if is_categorical:
                for label_val in unique_labels:
                    mask = self.labels == label_val
                    current_points_hover_texts = None
                    if point_hover_texts:
                        current_points_hover_texts = [ht for i, ht in enumerate(point_hover_texts) if mask[i]]

                    self.figure.add_trace(
                        go.Scatter(
                            x=self.data_2d[mask, 0],
                            y=self.data_2d[mask, 1],
                            mode="markers",
                            name=str(label_val),
                            hovertext=current_points_hover_texts,
                            hoverinfo=hoverinfo if current_points_hover_texts else "x+y+name",
                            marker={"size": 7},
                        ),
                    )
                self.figure.update_layout(legend_title_text=self.label_name, showlegend=True)
            else:  # Continuous labels
                self.figure.add_trace(
                    go.Scatter(
                        x=self.data_2d[:, 0],
                        y=self.data_2d[:, 1],
                        mode="markers",
                        hovertext=point_hover_texts,
                        hoverinfo=hoverinfo if point_hover_texts else "x+y", # 'z' for color data is implicit
                        marker={
                            "size": 7,
                            "color": self.labels,
                            "colorscale": "Viridis",
                            "showscale": True,
                            "colorbar": {"title": self.label_name},
                        },
                    ),
                )
                self.figure.update_layout(showlegend=False) # No legend needed if using a colorbar

    def show(self, width: int = 600, height: int = 600) -> None:
        self.figure.update_layout(
            width=width,
            height=height,
        )
        self.figure.show()

