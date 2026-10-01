"""Plot fuel sampling points and PWL interpolation in input and fuel-height views."""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from Functions.fuel_pwl_v1 import FuelFlowTable, KNOT_TO_MPS, FT_TO_M, FPM_TO_MPS


def plot_fuel_table(table):
    """Show descent, level-flight, and climb slices of the tetrahedral interpolant."""
    speed_axis, altitude_axis, vertical_axis = table.axes
    speed, altitude = np.meshgrid(
        np.unique(np.r_[speed_axis, np.linspace(speed_axis[0], speed_axis[-1], 61)]),
        np.unique(np.r_[altitude_axis, np.linspace(altitude_axis[0], altitude_axis[-1], 61)]),
        indexing="ij",
    )
    vertical_slices = (vertical_axis[0], 0.0, vertical_axis[-1])
    normalization = Normalize(vmin=table.flows.min(), vmax=table.flows.max())
    color_map = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(12, 9), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")

    for vertical in vertical_slices:
        flow = np.array([
            table.interpolate(horizontal, height, vertical)
            for horizontal, height in zip(speed.ravel(), altitude.ravel())
        ]).reshape(speed.shape)
        axis.plot_surface(
            speed / KNOT_TO_MPS,
            altitude / FT_TO_M,
            np.full_like(speed, vertical / FPM_TO_MPS),
            facecolors=color_map(normalization(flow)),
            rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False, alpha=0.3,
        )
        grid_speed, grid_altitude = np.meshgrid(
            speed_axis / KNOT_TO_MPS, altitude_axis / FT_TO_M, indexing="ij",
        )
        axis.plot_wireframe(
            grid_speed, grid_altitude,
            np.full_like(grid_speed, vertical / FPM_TO_MPS),
            rstride=1, cstride=1, color="#263238", linewidth=0.35, alpha=0.4,
        )

    axis.scatter(
        table.points[:, 0] / KNOT_TO_MPS,
        table.points[:, 1] / FT_TO_M,
        table.points[:, 2] / FPM_TO_MPS,
        c=table.flows, cmap=color_map, norm=normalization,
        s=16, edgecolors="black", linewidths=0.3, depthshade=False,
        label=f"{len(table.points)} OpenAP samples",
    )
    axis.legend(loc="upper left")
    axis.set_xlabel("Horizontal speed (knots)", labelpad=12)
    axis.set_ylabel("Altitude (ft)", labelpad=12)
    axis.set_zlabel("Vertical speed (ft/min)", labelpad=12)
    axis.set_xlim(speed_axis[0] / KNOT_TO_MPS, speed_axis[-1] / KNOT_TO_MPS)
    axis.set_ylim(altitude_axis[0] / FT_TO_M, altitude_axis[-1] / FT_TO_M)
    axis.set_zlim(vertical_axis[0] / FPM_TO_MPS, vertical_axis[-1] / FPM_TO_MPS)
    axis.set_zticks([vertical / FPM_TO_MPS for vertical in vertical_slices])
    axis.set_box_aspect((1.2, 1.2, 1))
    axis.view_init(elev=20, azim=-55)
    axis.set_title(
        f"OpenAP PWL fuel flow: {table.aircraft.upper()}, mass {table.mass_kg:,.0f} kg\n"
        "Sample points and translucent interpolated slices",
        pad=24,
    )
    figure.colorbar(
        ScalarMappable(norm=normalization, cmap=color_map), ax=axis,
        label="PWL fuel flow (kg/s)", shrink=0.65, pad=0.12,
    )
    return figure


def plot_fuel_surfaces(table):
    """Show exact triangular PWL surfaces at three sampled vertical speeds."""
    speed, altitude = np.meshgrid(
        table.axes[0] / KNOT_TO_MPS, table.axes[1] / FT_TO_M, indexing="ij",
    )
    indices = np.arange(speed.size).reshape(speed.shape)
    lower = indices[:-1, :-1].ravel()
    upper = indices[1:, 1:].ravel()
    triangles = np.vstack((
        np.column_stack((lower, indices[1:, :-1].ravel(), upper)),
        np.column_stack((lower, upper, indices[:-1, 1:].ravel())),
    ))
    mesh = Triangulation(speed.ravel(), altitude.ravel(), triangles)
    normalization = Normalize(vmin=table.flows.min(), vmax=table.flows.max())
    figure = plt.figure(figsize=(19, 7), layout="constrained")
    flows = table.flows.reshape(table.shape)
    vertical_indices = (0, len(table.axes[2]) // 2, len(table.axes[2]) - 1)
    for panel, vertical_index in enumerate(vertical_indices, start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        flow = flows[:, :, vertical_index].ravel()
        axis.plot_trisurf(
            mesh, flow, cmap="viridis", norm=normalization,
            edgecolor="#475569", linewidth=0.35, shade=False, alpha=0.9,
        )
        axis.scatter(
            speed.ravel(), altitude.ravel(), flow,
            color="black", s=9, depthshade=False, label=f"{flow.size} OpenAP samples",
        )
        axis.set_xlabel("Horizontal speed (knots)", labelpad=10)
        axis.set_ylabel("Altitude (ft)", labelpad=10)
        axis.set_zlabel("Fuel flow (kg/s)", labelpad=10)
        axis.set_zlim(table.flows.min(), table.flows.max())
        axis.set_title(f"Vertical speed: {table.axes[2][vertical_index] / FPM_TO_MPS:+.0f} ft/min")
        axis.set_box_aspect((1.2, 1.2, 1), zoom=0.82)
        axis.view_init(elev=25, azim=-55)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        f"PWL fuel surfaces: {table.aircraft.upper()}, mass {table.mass_kg:,.0f} kg\n"
        "Black dots: OpenAP samples | Triangular faces: linear interpolation",
        fontsize=15,
    )
    return figure


def plot_fuel_tetrahedron(table):
    """Show the complete tetrahedral grid with translucent interpolated face colors."""
    display_vertices = table.points / np.array([KNOT_TO_MPS, FT_TO_M, FPM_TO_MPS])
    faces = np.unique(np.sort(np.concatenate([
        table.simplices[:, face] for face in combinations(range(4), 3)
    ]), axis=1), axis=0)
    edges = np.unique(np.sort(np.concatenate([
        table.simplices[:, edge] for edge in combinations(range(4), 2)
    ]), axis=1), axis=0)
    normalization = Normalize(vmin=table.flows.min(), vmax=table.flows.max())
    color_map = plt.get_cmap("viridis")
    face_weights = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
    ])
    patches = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    face_points = face_weights @ display_vertices[faces]
    face_flows = table.flows[faces] @ face_weights.T
    patch_points = face_points[:, patches].reshape(-1, 3, 3)
    patch_flows = face_flows[:, patches].mean(axis=2).ravel()
    colors = color_map(normalization(patch_flows))
    colors[:, 3] = 0.035
    figure = plt.figure(figsize=(15, 11), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(Poly3DCollection(
        patch_points, facecolors=colors, edgecolors="none",
        linewidths=0, antialiased=False,
    ))
    axis.add_collection3d(Line3DCollection(
        display_vertices[edges], colors="black", linewidths=0.35, alpha=0.3,
    ))
    axis.scatter(
        *display_vertices.T, c=table.flows, cmap=color_map, norm=normalization,
        s=12, edgecolors="black", linewidths=0.35, depthshade=False,
    )
    axis.set_xlabel("Horizontal speed (knots)", labelpad=14)
    axis.set_ylabel("Altitude (ft)", labelpad=14)
    axis.set_zlabel("Vertical speed (ft/min)", labelpad=14)
    axis.set_xlim(display_vertices[:, 0].min(), display_vertices[:, 0].max())
    axis.set_ylim(display_vertices[:, 1].min(), display_vertices[:, 1].max())
    axis.set_zlim(display_vertices[:, 2].min(), display_vertices[:, 2].max())
    axis.set_box_aspect((1.2, 1.2, 1), zoom=0.9)
    axis.view_init(elev=22, azim=-55)
    figure.colorbar(
        ScalarMappable(norm=normalization, cmap=color_map), ax=axis,
        label="PWL fuel flow (kg/s), before transparency blending", shrink=0.65, pad=0.08,
    )
    figure.suptitle(
        f"Full PWL fuel mesh: {table.aircraft.upper()}, {len(table.points)} samples, "
        f"{len(table.simplices):,} tetrahedra\n"
        "Black outlines: tetrahedron edges | Translucent faces: interpolated fuel flow",
        fontsize=15,
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "Output" / "fuel_pwl_visualization" / "fuel_pwl_3d.png",
    )
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    table = FuelFlowTable()
    figure = plot_fuel_table(table)
    surfaces = plot_fuel_surfaces(table)
    tetrahedron = plot_fuel_tetrahedron(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    surface_output = args.output.with_name(args.output.stem + "_surfaces" + args.output.suffix)
    surfaces.savefig(surface_output, dpi=180)
    tetrahedron_output = args.output.with_name(args.output.stem + "_tetrahedron" + args.output.suffix)
    tetrahedron.savefig(tetrahedron_output, dpi=180)
    print(f"Saved PWL fuel plot: {args.output.resolve()}")
    print(f"Saved PWL fuel surfaces: {surface_output.resolve()}")
    print(f"Saved PWL fuel tetrahedron: {tetrahedron_output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(figure)
    plt.close(surfaces)
    plt.close(tetrahedron)


if __name__ == "__main__":
    main()