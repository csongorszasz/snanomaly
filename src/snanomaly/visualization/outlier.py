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
    model_params_display: dict[str, Any] = field()  # For display purposes

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
            "text": [f"SN: {name}, Score: {s:.4f}" for name, s in zip(self.sn_names, self.score_np, strict=True)],
            "hoverinfo": "text",
        }

        # Model-specific settings
        title_prefix = self.model_name
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

            # Define a list of (ax, ay) offsets to cycle through for annotations
            # distances = [33, 93, 125, 164, 192, 256, 299, 351, 398, 466, 489, 522]
            # angles_deg_priority = [90, 270, 0, 180, 45, 135, 225, 315]
            # angles_deg_general = [30, 60, 120, 150, 210, 240, 300, 330]
            distances = [50]
            angles_deg_priority = [0]
            angles_deg_general = [0]

            annotation_offsets = []

            # Generate priority offsets first
            for d in [35, 55]: # Slightly adjusted distances for priority
                for angle_deg in angles_deg_priority:
                    rad = np.deg2rad(angle_deg)
                    annotation_offsets.append({"ax": int(d * np.cos(rad)), "ay": -int(d * np.sin(rad))})

            # Generate general offsets
            for d in distances:
                for angle_deg in angles_deg_general:
                    rad = np.deg2rad(angle_deg)
                    # Add only if it's not too similar to an existing one (simple check)
                    new_offset = {"ax": int(d * np.cos(rad)), "ay": -int(d * np.sin(rad))}
                    if not any(abs(new_offset["ax"] - old["ax"]) < 10 and abs(new_offset["ay"] - old["ay"]) < 10 for old in annotation_offsets):
                        annotation_offsets.append(new_offset)

            # Fallback for very high number of outliers, add more variations
            if len(outlier_indices) > len(annotation_offsets):
                for d in [120, 140]:
                     for angle_deg in np.linspace(0, 360, 16, endpoint=False): # 16 angles
                        rad = np.deg2rad(angle_deg)
                        annotation_offsets.append({"ax": int(d * np.cos(rad)), "ay": -int(d * np.sin(rad))})


            for i, idx in enumerate(outlier_indices):
                current_offset = annotation_offsets[i % len(annotation_offsets)]
                self.figure.add_annotation(
                    x=x_vals[idx],
                    y=y_vals[idx],
                    text=self.sn_names[idx],
                    showarrow=True,
                    arrowhead=1,
                    arrowwidth=0.5,  # Thinner arrow
                    ax=current_offset["ax"],
                    ay=current_offset["ay"],
                    font={"size": 8},  # Smaller font
                    bgcolor="rgba(255,255,255,0.5)", # Slight background for text
                    borderpad=2, # Padding around text
                )

        # Layout and annotations
        plot_title = (
            f"{title_prefix}<br>"
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
