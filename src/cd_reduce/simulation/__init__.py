"""2D and 3D Born-approximation and Multislice simulations for CD-RSoXS."""

from cd_reduce.simulation.born import (
    BornSimulationResult2D,
    BornSimulationResult3D,
    simulate_born_2d,
    simulate_born_3d,
)
from cd_reduce.simulation.multislice import (
    MultisliceSimulationResult2D,
    MultisliceSimulationResult3D,
    simulate_multislice_2d,
    simulate_multislice_3d,
)
from cd_reduce.simulation.structures import (
    GratingLayerSpec,
    RefractiveIndexMap2D,
    RefractiveIndexMap3D,
    build_grating_2d,
    build_grating_3d,
    generate_rough_edge_1d,
)
from cd_reduce.simulation.visualization import (
    plot_born_vs_multislice_grid,
    plot_multislice_wavefield,
    plot_simulation_comparison,
    plot_structure_2d,
    plot_structure_3d,
)

__all__ = [
    "BornSimulationResult2D",
    "BornSimulationResult3D",
    "GratingLayerSpec",
    "MultisliceSimulationResult2D",
    "MultisliceSimulationResult3D",
    "RefractiveIndexMap2D",
    "RefractiveIndexMap3D",
    "build_grating_2d",
    "build_grating_3d",
    "generate_rough_edge_1d",
    "plot_born_vs_multislice_grid",
    "plot_multislice_wavefield",
    "plot_simulation_comparison",
    "plot_structure_2d",
    "plot_structure_3d",
    "simulate_born_2d",
    "simulate_born_3d",
    "simulate_multislice_2d",
    "simulate_multislice_3d",
]
