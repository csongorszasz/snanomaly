from __future__ import annotations

import pathlib
from typing import Any

import click
import numpy as np
import plotly.graph_objects as go
from attrs import define, field
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


@define
class PlotOutlier:
    features_2d_np: np.ndarray = field()
    pred_np: np.ndarray = field()
    score_np: np.ndarray = field()
    sn_names: list[str] = field()
    model_name: str = field()
    model_object: IsolationForest | OneClassSVM = field()
    dataset_name: str = field()
    dim_red_method: str = field()
    dims_str: str = field()
    model_params_display: dict[str, Any] = field() # For display purposes

    figure: go.Figure = field(factory=go.Figure, init=False)

    def __attrs_post_init__(self) -> None:
        self._generate_plot()

    def _generate_plot(self) -> None:
        x_vals, y_vals = self.features_2d_np[:, 0], self.features_2d_np[:, 1]
        num_outliers = np.sum(self.pred_np == -1)
        outlier_percentage = (num_outliers / len(self.pred_np)) * 100 if len(self.pred_np) > 0 else 0

        # Common scatter plot for all points
        scatter_trace_params: dict[str, Any] = {
            "x": x_vals,
            "y": y_vals,
            "mode": "markers",
            "text": [f"SN: {name}, Score: {s:.4f}" for name, s in zip(self.sn_names, self.score_np, strict=False)],
            "hoverinfo": "text",
        }

        # Model-specific settings
        if self.model_name == "IsolationForest":
            scatter_trace_params["marker"] = {
                "size": 5,
                "color": self.score_np,
                "colorscale": "Viridis",
                "colorbar": {"title": "Anomaly Score"},
                "line": {"width": 1, "color": "DarkSlateGrey"},
            }
            scatter_trace_params["name"] = "Data Points"
            outlier_marker_color = "red"
            title_prefix = "Isolation Forest"
        elif self.model_name == "OneClassSVM":
            scatter_trace_params["marker"] = {
                "size": 5,
                "color": self.score_np,
                "colorscale": "RdBu",
                "colorbar": {"title": "Decision Score"},
                "line": {"width": 1, "color": "DarkSlateGrey"},
            }
            scatter_trace_params["name"] = "Data Points"
            outlier_marker_color = "black"
            title_prefix = "OneClassSVM"
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        self.figure.add_trace(go.Scatter(**scatter_trace_params))

        # Highlight outliers
        outlier_indices = np.where(self.pred_np == -1)[0]
        if len(outlier_indices) > 0:
            self.figure.add_trace(go.Scatter(
                x=x_vals[outlier_indices],
                y=y_vals[outlier_indices],
                mode="markers",
                marker={"size": 12, "symbol": "circle-open", "color": outlier_marker_color},
                text=[f"SN: {self.sn_names[i]}, Score: {self.score_np[i]:.4f}" for i in outlier_indices],
                hoverinfo="text",
                name=f"Outliers ({num_outliers})",
            ))
            for idx in outlier_indices:
                self.figure.add_annotation(
                    x=x_vals[idx], y=y_vals[idx], text=self.sn_names[idx],
                    showarrow=True, arrowhead=1, yshift=10,
                )

        # Layout and annotations
        plot_title = (
            f"{title_prefix}: {self.dataset_name} ({self.dim_red_method} {self.dims_str})<br>"
            f"{num_outliers} outliers ({outlier_percentage:.2f}%)"
        )
        xaxis_title = "x"
        yaxis_title = "y"

        self.figure.update_layout(
            title=plot_title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="closest",
            template="plotly_white",
            width=1280,
            height=720,
        )

        # Model parameters annotation
        model_info_lines = [f"Model: {self.model_name}"]
        if self.model_name == "IsolationForest" and isinstance(self.model_object, IsolationForest):
            contam_val = self.model_params_display.get("contamination", "auto")
            if hasattr(self.model_object, "contamination_") and contam_val == "auto":
                contam_val = f"auto (fitted: {self.model_object.contamination_:.4f})"
            model_info_lines.extend([
                f"n_estimators: {self.model_params_display.get('n_estimators')}",
                f"contamination: {contam_val}",
                f"max_samples: {self.model_params_display.get('max_samples')}",
                f"max_features: {self.model_params_display.get('max_features')}",
                f"random_state: {self.model_params_display.get('random_state')}",
            ])
        elif self.model_name == "OneClassSVM" and isinstance(self.model_object, OneClassSVM):
            gamma_val = self.model_params_display.get("gamma", "scale")
            if hasattr(self.model_object, "gamma_") and isinstance(gamma_val, str):
                gamma_val = f"{gamma_val} (fitted: {self.model_object.gamma_:.4f})"
            model_info_lines.extend([
                f"kernel: {self.model_params_display.get('kernel')}",
                f"nu: {self.model_params_display.get('nu'):.4f}",
                f"gamma: {gamma_val}",
            ])
            if self.model_params_display.get("kernel") == "poly":
                model_info_lines.append(f"degree: {self.model_params_display.get('degree')}")
            if self.model_params_display.get("kernel") in ["poly", "sigmoid"]:
                model_info_lines.append(f"coef0: {self.model_params_display.get('coef0')}")

        self.figure.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.01, text="<br>".join(model_info_lines),
            showarrow=False, align="left", bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1,
        )

        self.figure.update_xaxes(scaleanchor="y", scaleratio=1)
        self.figure.update_yaxes(constrain="domain")

    def write_image(self, path: pathlib.Path, width: int = 1280, height: int = 720) -> None:
        self.figure.update_layout(width=width, height=height)
        self.figure.write_image(str(path))
        click.echo(f"Plot saved to `{path}`")

    def show(self, width: int = 1280, height: int = 720) -> None:
        self.figure.update_layout(width=width, height=height)
        self.figure.show()

