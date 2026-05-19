from sympy import *
from typing import Dict, List, Set, Tuple

# finding close parens
def find_parens(s: str) -> Dict[int, int]:
    # source: https://stackoverflow.com/a/29992065
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == "(":
            pstack.append(i)
        elif c == ")":
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def swap_fn(s: str, swaps: List[Tuple[str, str]]) -> str:
    parens = find_parens(s)
    new = list(s)
    newstr = s
    for (tofind, toreplace) in swaps:
        while newstr.find(tofind) > -1:
            lendiff = len(toreplace) - len(tofind)
            first_idx = newstr.find(tofind)
            new = list(newstr)
            new[first_idx : first_idx + len(tofind)] = toreplace
            open_idx = newstr.find("(", first_idx)
            # account for different operator length
            new[open_idx + lendiff] = "["
            close_idx = parens[open_idx]
            # account for different operator length
            new[close_idx + lendiff] = "]"
            newstr = "".join(new)
    return newstr


def swap_piecewise(s: str):
    newstr = s
    while newstr.find("Piecewise(") > -1:
        parens = find_parens(newstr)
        val = newstr.find("Piecewise(") + len("Piecewise(") - 1
        newstr = newstr[:val] + "[{" + newstr[val + 1 :]
        newstr = newstr[: parens[val] + 1] + "}]" + newstr[parens[val] + 2 :]
        a = newstr[val + 2 : parens[val] + 1]  # input to Piecewise
        newa = []
        count = 0
        for (i, c) in enumerate(a):
            if c == "(":
                count += 1
                if count == 1:
                    newa.append("{")
                else:
                    newa.append("(")
            elif c == ")":
                count -= 1
                if count == 0:
                    newa.append("}")
                else:
                    newa.append(")")
            else:
                newa.append(c)

        a = "".join(newa)
        newstr = newstr[: val + 2] + a + newstr[parens[val] + 1 :]
    return newstr


def sympy_to_mathematica(cond) -> str:
    cond_string: str = str(cond)
    replacements = [
        ("**", "^"),
        ("|", "||"),
        ("&", "&&"),
        ("pi", "Pi"),
    ]
    for swap in replacements:
        assert (
            len(swap) == 2
        ), "There must be only two elements for specifying replacements"
        cond_string = cond_string.replace(swap[0], swap[1])

    functions = [
        ("sin", "Sin"),
        ("cos", "Cos"),
        ("tan", "Tan"),
        ("sqrt", "Sqrt"),
        # ("Abs", "Abs"),
        # ("exp", "E"),  # maybe not the case
    ]

    cond_string = swap_fn(cond_string, functions)

    cond_string = swap_piecewise(cond_string)

    return cond_string


def print_mathematica(
    x, y, cond, xbounds, ybounds, trajectory, poly, print_command=False
):
    if y not in trajectory.free_symbols:
        func_var = x
    elif x not in trajectory.free_symbols:
        func_var = y
    else:
        raise Exception("Trajectory had two variables!")

    # Translating to mathematica here for one-click plotting
    # boolean safe region condition for RegionPlot
    cond_mathematica: str = sympy_to_mathematica(cond)
    # f(x) works with Plot[], f(y) needs contour plot
    region_plotrange_mathematica: str = (
        f"{{x, {xbounds[0]}, {xbounds[1]}}}, {{y, {ybounds[0]}, {ybounds[1]}}}"
    )
    if func_var == x:
        traj_plotrange_mathematica: str = f"{{x, {xbounds[0]}, {xbounds[1]}}}"
        traj_mathematica: str = sympy_to_mathematica(trajectory)
        plot_type = "Plot"
        style_type = "PlotStyle"
        offsetx = (xbounds[1] + xbounds[0]) / 2
        offsety = trajectory.subs(x, offsetx)
    elif func_var == y:
        traj_plotrange_mathematica: str = region_plotrange_mathematica
        traj_mathematica: str = sympy_to_mathematica(-x + trajectory) + " == 0"
        plot_type = "ContourPlot"
        style_type = "ContourStyle"
        offsety = (ybounds[1] + ybounds[0]) / 2
        offsetx = trajectory.subs(y, offsety)

    verts = poly.vertices
    mathematica_vertices = str([(v.x + offsetx, v.y + offsety) for v in verts])
    mathematica_vertices = "{" + sympy_to_mathematica(mathematica_vertices)[1:-1] + "}"
    mathematica_vertices = (
        "Polygon[" + mathematica_vertices.replace("(", "{").replace(")", "}") + "]"
    )

    TRAJ_COLOR = "Purple"
    POLY_COLOR = "Green"
    mathematica_header = (
        "\n======== Output for Plotting Safe Region in Mathematica ========\n"
    )
    mathematica_footer = (
        "\n=========================== Copy me! ===========================\n"
    )
    mathematica_output = f"""\nShow[
    RegionPlot[{cond_mathematica},  {region_plotrange_mathematica}, PlotPoints -> 60,  MaxRecursion -> 5],
    {plot_type}[{traj_mathematica},  {traj_plotrange_mathematica}, {style_type}->{{{TRAJ_COLOR}, Dashed}}],
    Graphics[ {{FaceForm[None], EdgeForm[{POLY_COLOR}], {mathematica_vertices} }} ],
    GridLines->Automatic, Ticks->Automatic, AspectRatio->Automatic\n]\n"""

    if print_command:
        print(mathematica_header, mathematica_output, mathematica_footer)

    return mathematica_output
