from __future__ import annotations

from enum import Enum

from plotly.colors import qualitative


class Color(Enum):
    B = "blue"
    R = "magenta"
    I = "goldenrod"
    g = "green"
    r = "red"
    i = "brown"
    g_pr = qualitative.Pastel[8]
    r_pr = qualitative.Pastel[6]
    i_pr = qualitative.Pastel[7]
