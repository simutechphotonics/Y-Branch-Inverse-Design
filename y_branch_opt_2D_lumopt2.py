#############################################################################
# Scriptfile: y_branch_opt_2D_lumopt2.py
#
# Description:
# lumopt2 port of y_branch_opt_2D.py / y_branch_opt_3D.py: adjoint shape
# optimization for the SOI Y-branch, using a real FOM combining the two
# output waveguide port transmissions into a single 50/50-split objective.
#
# Notes / deliberate deviations from the original lumopt scripts:
# - lumopt2's FdtdSession only supports real (3D) FDTD -- it does not
#   support MODE/varFDTD, and a live test showed 2D FDTD ("dimension":
#   "2D") crashes inside lumopt2.core internals (store_mesh_info assumes
#   a real z-mesh array). This script is therefore built on full 3D FDTD,
#   like y_branch_opt_3D.py, rather than the 2D varFDTD original.
# - The original scripts placed a single combined "fom" monitor spanning
#   both output waveguides and used ModeMatch against the fundamental
#   supermode. This script instead uses two independent FDTD ports (one
#   per output branch, each spanning half the y-domain split at y=0) and
#   combines their transmissions directly, per the user's explicit request
#   for a literal 50/50-split FOM between the two output transmissions.
# - No symmetry boundary conditions are used (all-PML boundaries), since
#   the two-port FOM no longer relies on the symmetric/anti-symmetric
#   supermode trick the original combined-monitor approach used.
# - Single wavelength (1550 nm) rather than the original 1300-1800 nm/21
#   point sweep, to keep each FDTD run fast for testing.
##############################################################################

import os
import sys
import numpy as np
import scipy as sp
from scipy import interpolate

# lumopt2 ships inside the Lumerical install, not on the default Python path.
sys.path.insert(0, r"C:\Program Files\Lumerical\v261\api\python")
import lumopt2 as lmpt

######## SPECTRAL POINT ########
center_wavelength = 1550e-9

######## OPTIMIZABLE GEOMETRY ########
# Same 10-parameter cubic-spline taper as the original lumopt scripts.
initial_points_x = np.linspace(-1.0e-6, 1.0e-6, 10)
initial_points_y = np.linspace(0.25e-6, 0.6e-6, initial_points_x.size)


def splitter_vertices(params):
    """Compute the closed polygon vertices for the spline-defined splitter taper."""
    points_x = np.concatenate(([initial_points_x.min() - 0.01e-6], initial_points_x, [initial_points_x.max() + 0.01e-6]))
    points_y = np.concatenate(([initial_points_y.min()], params, [initial_points_y.max()]))
    n_interpolation_points = 100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind="cubic")
    polygon_points_y = interpolator(polygon_points_x)
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    return np.array(polygon_points_up[::-1] + polygon_points_down)


def splitter_param_map(params):
    """Map spline y-coordinates to the 'splitter' polygon's vertices property."""
    return {"splitter::vertices": splitter_vertices(params)}


bounds = [(0.2e-6, 0.8e-6)] * initial_points_y.size
depth = 220.0e-9


######## BASE SIMULATION ########
def y_branch_setup(fdtd):
    """Build the base 3D FDTD Y-branch simulation (materials, geometry, ports, mesh)."""
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()

    size_x = 3e-6
    size_y = 3e-6
    size_z = 1.2e-6
    mesh_x = 20e-9
    mesh_y = 20e-9
    mesh_z = 20e-9
    finer_mesh_size = 2.5e-6
    finer_mesh_size_z = 0.6e-6
    mesh_accuracy = 4

    ## MATERIALS (non-dispersive, fixed at center wavelength -- matches original)
    opt_material = fdtd.addmaterial("Dielectric")
    fdtd.setmaterial(opt_material, "name", "Si: non-dispersive")
    n_opt = fdtd.getindex("Si (Silicon) - Palik", 299792458.0 / center_wavelength)
    fdtd.setmaterial("Si: non-dispersive", "Refractive Index", n_opt)

    sub_material = fdtd.addmaterial("Dielectric")
    fdtd.setmaterial(sub_material, "name", "SiO2: non-dispersive")
    n_sub = fdtd.getindex("SiO2 (Glass) - Palik", 299792458.0 / center_wavelength)
    fdtd.setmaterial("SiO2: non-dispersive", "Refractive Index", n_sub)
    fdtd.setmaterial("SiO2: non-dispersive", "color", np.array([0, 0, 0, 0]))

    ## GEOMETRY
    fdtd.addrect({"name": "input wg", "x": -2.5e-6, "x span": 3e-6, "y": 0, "y span": 0.5e-6, "z": 0, "z span": depth, "material": "Si: non-dispersive"})
    fdtd.addrect({"name": "output wg top", "x": 2.5e-6, "x span": 3e-6, "y": 0.35e-6, "y span": 0.5e-6, "z": 0, "z span": depth, "material": "Si: non-dispersive"})
    fdtd.addrect({"name": "output wg bottom", "x": 2.5e-6, "x span": 3e-6, "y": -0.35e-6, "y span": 0.5e-6, "z": 0, "z span": depth, "material": "Si: non-dispersive"})
    fdtd.addrect({"name": "sub", "x": 0, "x span": 8e-6, "y": 0, "y span": 8e-6, "z": 0, "z span": 10e-6, "material": "SiO2: non-dispersive", "override mesh order from material database": 1, "mesh order": 3, "alpha": 0.8})

    # Optimizable "splitter" polygon, initialized with the default spline shape.
    fdtd.addpoly({"name": "splitter", "vertices": splitter_vertices(initial_points_y), "z": 0, "z span": depth, "material": "Si: non-dispersive"})

    ## FDTD REGION
    fdtd.addfdtd({
        "dimension": "3D",
        "mesh accuracy": mesh_accuracy,
        "x min": -size_x / 2, "x max": size_x / 2,
        "y min": -size_y / 2, "y max": size_y / 2,
        "z min": -size_z / 2, "z max": size_z / 2,
    })

    ## GLOBAL SOURCE / MONITOR SETTINGS (single wavelength)
    fdtd.setglobalsource("wavelength start", center_wavelength)
    fdtd.setglobalsource("wavelength stop", center_wavelength)
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("frequency points", 1)

    ## PORTS
    fdtd.addport({"name": "input_port", "injection axis": "x-axis", "direction": "Forward", "mode selection": "fundamental TE mode", "x": -1.23e-6, "y": 0, "y span": size_y, "z": 0, "z span": size_z})
    fdtd.addport({"name": "output_top", "injection axis": "x-axis", "direction": "Backward", "mode selection": "fundamental TE mode", "x": 1.23e-6, "y": size_y / 4, "y span": size_y / 2, "z": 0, "z span": size_z})
    fdtd.addport({"name": "output_bottom", "injection axis": "x-axis", "direction": "Backward", "mode selection": "fundamental TE mode", "x": 1.23e-6, "y": -size_y / 4, "y span": size_y / 2, "z": 0, "z span": size_z})
    fdtd.setnamed("FDTD::ports", "source port", "input_port")
    fdtd.setnamed("FDTD::ports", "source mode", "mode 1")

    ## MESH OVERRIDE IN OPTIMIZABLE REGION
    fdtd.addmesh({"name": "opt_mesh", "x": 0, "x span": finer_mesh_size, "y": 0, "y span": finer_mesh_size, "z": 0, "z span": finer_mesh_size_z, "dx": mesh_x, "dy": mesh_y, "dz": mesh_z})

    ## 2D FIELD MONITOR FOR LIVE VISUALIZATION (not the 3D "optimization_dft" monitor,
    ## which MonitorPanel can't auto-plot since it's a true (x,y,z) volume, not an image).
    fdtd.adddftmonitor({"name": "field_monitor", "monitor type": "2D Z-normal", "x": 0, "x span": size_x, "y": 0, "y span": size_y, "z": 0})


######## FIGURE OF MERIT: 50/50 SPLIT BETWEEN THE TWO OUTPUT PORTS ########
def balanced_split_fom(sim_results):
    """Maximized when output_top and output_bottom transmissions are equal (50/50 split)."""
    T_top = sim_results[0]
    T_bottom = sim_results[1]
    total_T = T_top + T_bottom
    imbalance = (T_top - T_bottom) ** 2
    return total_T - imbalance


port_top_result = lmpt.PortResults(monitor_name="output_top", metric="transmission", wavelengths=center_wavelength)
port_bottom_result = lmpt.PortResults(monitor_name="output_bottom", metric="transmission", wavelengths=center_wavelength)
fom = lmpt.Fom([port_top_result, port_bottom_result], fct=balanced_split_fom)


######## PARAMETRIZATION ########
# scipy's cubic spline interpolation is not autograd-differentiable, so use_jac=False
# (lumopt2 falls back to a conservative "every parameter affects every pixel" dEps assumption).
optimization_region = lmpt.Box(
    x_span=2.5e-6, y_span=2.5e-6, z_span=depth,
    dx=20e-9, dy=20e-9, dz=20e-9,
)
parametrization = lmpt.Parametrization(
    func=splitter_param_map,
    bounds=bounds,
    optimization_region=optimization_region,
    initial_params=initial_points_y,
    use_jac=False,
)

######## OPTIMIZER ########
optimizer = lmpt.ScipyOptimizer(
    method="L-BFGS-B",
    max_iter=30,  # small test run; original used 30
    ftol=1e-5,
    gtol=1e-5,
)

######## PROJECT ########
project = lmpt.Project(
    setup=y_branch_setup,
    parametrization=parametrization,
    fom=fom,
    project_name="y_branch_2D_lumopt2",
)


######## GEOMETRY VISUALIZATION ########
# Unlike ClosedCurve, the generic Parametrization class has no built-in
# visualize(ax, params, initial_params), so GeometryPanel needs an explicit
# adapter that knows how to turn spline params into a drawable outline.
class SplitterGeometryVisualizer:
    def visualize(self, ax, params, initial_params):
        if initial_params is not None:
            baseline = splitter_vertices(initial_params)
            ax.plot(baseline[:, 0] * 1e6, baseline[:, 1] * 1e6, color="0.7", linestyle="--", label="initial")
        verts = splitter_vertices(params)
        ax.plot(verts[:, 0] * 1e6, verts[:, 1] * 1e6, color="tab:blue", label="current")
        ax.legend(loc="upper right", fontsize="small")


splitter_geometry_visualizer = SplitterGeometryVisualizer()


######## VISUALIZER ########
# Matches the officially-documented lumopt2 Y-branch example's visualizer
# (https://lumerical.docs.pyansys.com/.../pylumerical_y_branch.html):
# FOM trace, gradient-norm trace, current geometry (via the parametrization's
# own `visualize()`, called by GeometryPanel), and live |T_out| per port
# (the exact quantity the FOM is built from).
class ProgressPlotVisualizer(lmpt.GraphicalVisualizer):
    """GraphicalVisualizer that saves its PNGs into a 'progress plot' subfolder."""

    def on_optimization_start(self, project, num_params, bounds, **kwargs):
        super().on_optimization_start(project, num_params, bounds, **kwargs)
        if self.save_dir is not None:
            self.save_dir = self.save_dir / "progress plot"
            self.save_dir.mkdir(parents=True, exist_ok=True)


visualizer = ProgressPlotVisualizer(
    figsize=(12, 10),
    layout=(2, 3),
    panels=[
        lmpt.FomPanel(),
        lmpt.GradientNormPanel(),
        lmpt.GeometryPanel(parametrization=splitter_geometry_visualizer),
        lmpt.MonitorPanel(
            monitor_name="FDTD::ports::output_top",
            result_name="expansion for port monitor.T_out",
            operation="abs",
            title="|T_out| output_top (matches FOM)",
            axes_kwargs={"ylim": (0.0, 1.0)},
        ),
        lmpt.MonitorPanel(
            monitor_name="FDTD::ports::output_bottom",
            result_name="expansion for port monitor.T_out",
            operation="abs",
            title="|T_out| output_bottom (matches FOM)",
            axes_kwargs={"ylim": (0.0, 1.0)},
        ),
        lmpt.MonitorPanel(
            monitor_name="field_monitor",
            result_name="E",
            operation="abs^2",
            title="Forward Fields E^2",
        ),
    ],
)

optimization = lmpt.Optimization(
    project=project,
    optimizer=optimizer,
    callbacks=[lmpt.FileLogger(), visualizer],
    store_all_simulations=False,
    log_profiling_summary=True,
)

def export_design(params, filename):
    """Build the Y-branch with the given spline params and save it as a standalone .fsp."""
    import lumapi
    with lumapi.FDTD(hide=True) as fdtd:
        y_branch_setup(fdtd)
        fdtd.setnamed("splitter", "vertices", splitter_vertices(params))
        fdtd.save(filename)


if __name__ == "__main__":
    optimal_params, final_fom = optimization.run()
    print("Optimal params:", optimal_params)
    print("Final FOM:", final_fom)
    np.savetxt(os.path.join(os.path.dirname(__file__), "2D_parameters_lumopt2.txt"), optimal_params)

    here = os.path.dirname(__file__)
    export_design(initial_points_y, os.path.join(here, "y_branch_2D_lumopt2_INITIAL.fsp"))
    print("Exported initial design to y_branch_2D_lumopt2_INITIAL.fsp")    
    export_design(optimal_params, os.path.join(here, "y_branch_2D_lumopt2_FINAL.fsp"))
    print("Exported final design to y_branch_2D_lumopt2_FINAL.fsp")
