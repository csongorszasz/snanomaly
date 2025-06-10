class FigLayout:
    @classmethod
    def _xaxis_mjd(cls) -> dict:
        return {
            "title": {
                "text": "Time (Modified Julian Date)",
                "font": {
                    "size": 12,
                },
            },
            "exponentformat": "none",
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "ticks": "inside",
            "minor": {
                "ticks": "inside",
                "showgrid": True,
            },
        }

    @classmethod
    def _yaxis_flux(cls) -> dict:
        return {
            "title": {
                "text": r"$\text{Flux }(erg\,s^{-1}\,Hz^{-1}\,cm^{-2})$",
                "font": {
                    "size": 12,
                },
            },
            "exponentformat": "power",
            "showexponent": "all",
            "linewidth": 1,
            "linecolor": "black",
            "mirror": True,
            "ticks": "inside",
            "minor": {
                "ticks": "inside",
                "showgrid": True,
            },
        }

    @classmethod
    def _legend_top_right(cls) -> dict:
         return {
            "yanchor": "top",
            "xanchor": "right",
        }

    @classmethod
    def _legend_bottom_horizontal(cls):
        pass

    @classmethod
    def _legend_bands(cls):
        return {
            "bordercolor": "black",
            "borderwidth": 1,
            "orientation": "h",
            "xanchor": "right",
            "x": 0.98,
            "yanchor": "top",
            "y": 0.98,
            "itemwidth": 30,
            "font": {
                "size": 8,
            },
        }

    @classmethod
    def _template_light(cls) -> str:
        return "plotly_white"

    @classmethod
    def _template_dark(cls) -> str:
        return "plotly_dark"

    @classmethod
    def light_curves(cls) -> dict:
        return {
            "xaxis": cls._xaxis_mjd(),
            "yaxis": cls._yaxis_flux(),
            "template": cls._template_light(),
            "legend": cls._legend_bands(),
        }
