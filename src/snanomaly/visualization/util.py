from webcolors import name_to_rgb


def color_to_rgba(color: str, alpha: float) -> str:
    if "rgb" in color:
        rgb_values_str = color[4:-1].split(",")
        r, g, b = map(int, rgb_values_str)
    else:
        color = name_to_rgb(color)
        r, g, b = color.red, color.green, color.blue
    return f"rgba({r}, {g}, {b}, {alpha})"
