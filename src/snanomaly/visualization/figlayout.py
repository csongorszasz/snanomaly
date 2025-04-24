class FigLayout:
    @classmethod
    def _xaxis_mjd(cls) -> dict:
        return {"title_text": "Time (Modified Julian Date)", "exponentformat": "none"}

    @classmethod
    def _yaxis_flux(cls) -> dict:
        return {"title_text": "Flux", "exponentformat": "power", "showexponent": "all"}

    @classmethod
    def _legend_top_right(cls) -> dict:
         return {
            "yanchor": "top",
            "xanchor": "right",
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
            "legend": cls._legend_top_right(),
        }
