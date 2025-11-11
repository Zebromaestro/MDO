# Final MDO for Magpie
# Code written by Ryan Mattana and Inigo Perez at Rice University
# The goal of this code is to explore the design space for the following geometrical parameters:
# alpha at cruise; boom length; tail position; fuselage position; wing span; wing_root_chord;
# wing taper ratio; winglet height fraction; winglet cant angle; winglet sweep angle; winglet taper ratio;
# vtail span; vtail root chord; vtail taper ratio; vtail dihedral; vtail le sweep; incident tail degree; engine span fraction;
# wing motor x frac; all of the point masses of our electronics;
# the discrete choices wing_airfoil_name and vtail_airfoil_name.


# This code uses uses low-speed, quasi-steady, incompressible aerodynamics via AeroSandbox’s AeroBuildup and cross-checks
# with AVL (vortex-lattice, linear potential flow). The code functions by building a 3D wing + V-tail + pod-and-boom geometry,
# pulls 2D airfoil polars (if available) and generates them using XFoil. Trim is enforced for Cm=0 and we constrain the
# lift for to be equal to the weight. The landscape of geometries is constraine by literature values for stability-derivatives.
# mass properties are built from simple structural models (skin areal densities, spar linear densities, boom linear density)
# plus point masses (motors, gear, payload, electronics), summed to TOGW for weight and CG. Our competition constrained it to
# 3.5 lbs.

import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from pathlib import Path
import itertools
import traceback
import casadi as cas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import aerosandbox.tools.pretty_plots as p
import os
from datetime import datetime

timestamp = datetime.now().strftime("20251019_035712")  # e.g., 20251019_035712

# Add AVL directory to PATH temporarily for this session
os.environ["PATH"] = r"C:\Users\inhip\Documents\Rice_Flight;" + os.environ["PATH"]


# ----------------- Tunables -----------------
RHO = 1.225                 # kg/m^3
CRUISE_TO_VSTALL = 1.15     # Vcruise = 1.15 * Vstall

MAKE_PLOTS = True
AIRFOILS_WING = ["e210"]
AIRFOILS_VTAIL = ["naca0009"]
CONFIG_NAME_BASE = "MDO_1010_PodBoomSweep"

# Objective weights/targets
LD_WEIGHT              = 0.0667
SINK_WEIGHT            = 2
STALL_WEIGHT           = 4
E_OSWALD               = 0.85
INDUCED_PENALTY_WEIGHT = 20
MASS_WEIGHT = 20

# CLmax estimation fallback table
WING_CLMAX_LOOKUP = {
    "ag13": 1.2,
    "ag09": 1.3,
    "ag04": 1.1,
    "s1223": 1.8,
}
CLMAX_MARGIN = 0.90

# ---- Structural mass models ----
WING_SKIN_AREAL_DENS    = 1.4796
VTAIL_SKIN_AREAL_DENS   = 1.4796


print("Performing common setup...")

def get_save_directory(preferred: Path = None) -> Path:
    if preferred is not None:
        try:
            preferred.drive
            if preferred.drive and not Path(preferred.drive + '\\').exists():
                raise FileNotFoundError(f"Drive {preferred.drive} not found.")
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except Exception:
            pass
    fallback = Path(__file__).resolve().parent / "figures"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

preferred_dir = Path(r"C:\Users\matta\PycharmProjects\MDO\Figures")
save_directory = get_save_directory(preferred=preferred_dir)
print(f"Figures will be saved to: {save_directory.resolve()}")

# --- Airfoils + polars
def load_airfoils(names):
    afs = {}
    for name in names:
        af = asb.Airfoil(name=name)
        try:
            af.generate_polars(cache_filename=f"cache/{name}.json", alphas=np.linspace(-10, 20, 21))
            print(f"Generated polars for {name}")
        except Exception as e:
            print(f"Could not generate polars for {name}: {e}")
        afs[name] = af
    return afs

airfoil_names_needed = list(set(AIRFOILS_WING + AIRFOILS_VTAIL + ["ag04", "ag09", "ag13", "naca0008"]))
airfoils = load_airfoils(airfoil_names_needed)

# ---------- Helper Functions ----------
def _three_view_to_array(airplane, dpi: int = 200):
    plt.ioff()
    airplane.draw_three_view(show=False)
    fig = plt.gcf()
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)
    plt.close(fig)
    return img

def print_title(s: str):
    print("\n" + "*" * 25 + f" {s.upper()} " + "*" * 25 + "\n")

def _composite_report_save(config_name, airplane, op_point, mass_props, mass_props_TOGW, aero, sol, dpi=200):
    import time
    alpha_range = np.linspace(-15, 25, 180)
    aero_polars = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=op_point.velocity, alpha=alpha_range),
        xyz_ref=mass_props_TOGW.xyz_cg
    ).run()

    fig = plt.figure(figsize=(24, 18), dpi=dpi, constrained_layout=True)
    fig.suptitle(f"{config_name.replace('_', ' ')}: L/D and Stability Report", fontsize=24, y=0.98)
    gs = fig.add_gridspec(3, 3)

    # Three-view + mass pie
    ax_plane = fig.add_subplot(gs[0, 0:2])
    try:
        tv_img = _three_view_to_array(airplane, dpi=max(200, dpi))
        ax_plane.imshow(tv_img)
        ax_plane.set_title("Airplane (Three-View Rendering)", fontsize=16)
        ax_plane.axis("off")
    except Exception as e:
        ax_plane.text(0.5, 0.5, f"three_view render failed:\n{e}", ha="center", va="center")
        ax_plane.axis("off")

    ax_pie = fig.add_subplot(gs[0, 2])
    name_remaps = {k: k.replace("_", " ").title() for k in mass_props.keys()}
    mass_props_to_plot = {k: v for k, v in mass_props.items() if float(v.mass) > 1e-9}
    if "ballast" in mass_props_to_plot and float(mass_props_to_plot["ballast"].mass) < 1e-6:
        del mass_props_to_plot["ballast"]
    plt.sca(ax_pie)
    p.pie(
        values=[v.mass for v in mass_props_to_plot.values()],
        names=[name_remaps.get(n, n) for n in mass_props_to_plot.keys()],
        center_text=(f"Mass Budget\nTOGW: {mass_props_TOGW.mass * 1e3:.2f} g"),
        startangle=110, arm_length=28, arm_radius=18, y_max_labels=1.10
    )
    ax_pie.set_title("Mass Budget", fontsize=16, pad=16)

    # CL, CD, L/D
    ax_cl = fig.add_subplot(gs[1, 0]); ax_cd = fig.add_subplot(gs[1, 1]); ax_ld = fig.add_subplot(gs[1, 2])
    ax_cl.plot(alpha_range, aero_polars["CL"])
    ax_cd.plot(alpha_range, aero_polars["CD"])
    ax_ld.plot(alpha_range, aero_polars["CL"] / (aero_polars["CD"] + 1e-12))

    ax_cl.set_xlabel("α [deg]"); ax_cl.set_ylabel("CL"); ax_cl.set_title("CL vs α"); ax_cl.grid(True)
    ax_cd.set_xlabel("α [deg]"); ax_cd.set_ylabel("CD"); ax_cd.set_title("CD vs α"); ax_cd.grid(True); ax_cd.set_ylim(bottom=0)
    ax_ld.set_xlabel("α [deg]"); ax_ld.set_ylabel("L/D"); ax_ld.set_title("L/D vs α"); ax_ld.grid(True)

    # SM + Cm + convergence
    ax_sm = fig.add_subplot(gs[2, 0]); ax_cm = fig.add_subplot(gs[2, 1]); ax_feas = fig.add_subplot(gs[2, 2])
    x_np = aero["x_np"]
    mac = airplane.wings[0].mean_aerodynamic_chord()
    static_margin_actual = (x_np - mass_props_TOGW.x_cg) / mac
    x_cg_range = np.linspace(x_np - 0.5 * mac, x_np + 0.5 * mac, 200)
    sm_range = (x_np - x_cg_range) / mac
    ax_sm.plot(x_cg_range, sm_range * 100)
    ax_sm.axvline(mass_props_TOGW.x_cg, color='r', linestyle='--',
                  label=f"Actual CG\nSM = {static_margin_actual * 100:.1f}%")
    ax_sm.axhline(0, color='k', linewidth=0.8)
    ax_sm.set_xlabel("CG Location x_cg [m]")
    ax_sm.set_ylabel("Static Margin [% MAC]")
    ax_sm.set_title("Static Margin vs. CG Location"); ax_sm.grid(True); ax_sm.legend()

    ax_cm.plot(alpha_range, aero_polars["Cm"])
    ax_cm.set_xlabel("α [deg]"); ax_cm.set_ylabel("Cm"); ax_cm.set_title("Cm vs α"); ax_cm.grid(True)
    ax_cm.axhline(0, color='k', lw=0.8)
    cm_vals = aero_polars["Cm"]
    pad = max(0.12, 0.2 * float(np.max(np.abs(cm_vals)) + 1e-6))
    ax_cm.set_ylim(-pad, +pad)

    try:
        stats = sol.stats(); it = stats.get("iterations", {}); inf_pr = it.get("inf_pr", []); inf_du = it.get("inf_du", [])
        if inf_pr and inf_du:
            ax_feas.semilogy(inf_pr, label="Primal Feasibility"); ax_feas.semilogy(inf_du, label="Dual Feasibility")
            ax_feas.set_xlabel("Solver Iteration"); ax_feas.set_ylabel("Feasibility (log)")
            ax_feas.set_title("Convergence History"); ax_feas.grid(True, which='both', linestyle=':'); ax_feas.legend()
        else:
            ax_feas.text(0.5, 0.5, "Solver stats not available.", ha="center", va="center"); ax_feas.axis('off')
    except Exception as e:
        ax_feas.text(0.5, 0.5, f"Could not plot feasibility:\n{e}", ha="center", va="center"); ax_feas.axis('off')

    base = save_directory / f"{config_name}_report.png"
    ts = __import__('time').strftime("%Y%m%d_%H%M%S")
    stamp = save_directory / f"{config_name}_report__{ts}.png"
    for out in (base, stamp):
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        try:
            with open(out, "rb") as f:
                size = len(f.read())
            print(f"Saved report figure to: {out.resolve()}  ({size/1024:.1f} kB)")
        except Exception:
            print(f"Saved report figure to: {out.resolve()}")
    plt.close(fig); plt.close("all")

def _plot_mass_distribution(config_name, mass_props, mass_props_TOGW, boom_nose_x, x_tail, wing_ac, x_np, dpi=200):
    fig, ax = plt.subplots(figsize=(16, 4), dpi=dpi)
    ax.plot([boom_nose_x, x_tail], [0, 0], color='black', linewidth=3, zorder=1)
    ax.scatter([boom_nose_x, x_tail], [0, 0], color='black', s=100, zorder=1)
    ax.text(boom_nose_x, 0.03, "Boom Start", ha='center', va='bottom', fontsize=10)
    ax.text(x_tail, 0.03, "Tail Start", ha='center', va='bottom', fontsize=10)

    y_offset_alternator = 1
    for name, props in mass_props.items():
        mass = float(props.mass)
        if mass < 1e-6:
            continue
        x_cg = props.x_cg[0] if hasattr(props.x_cg, '__iter__') else float(props.x_cg)
        marker_size = 50 + mass * 1500
        ax.scatter(x_cg, 0, s=marker_size, alpha=0.6, zorder=2, label=f"{name} ({mass * 1e3:.1f} g)")
        y_text = 0.08 * y_offset_alternator
        ax.plot([x_cg, x_cg], [0, y_text], color='gray', linestyle='--', linewidth=0.8)
        ax.text(x_cg, y_text + 0.01, name.replace('_', ' ').title(), ha='center',
                va='bottom' if y_offset_alternator > 0 else 'top', fontsize=9)
        y_offset_alternator *= -1

    cg_x_total = mass_props_TOGW.xyz_cg[0] if hasattr(mass_props_TOGW.xyz_cg, '__iter__') else mass_props_TOGW.xyz_cg
    ax.axvline(cg_x_total, color='red', linestyle=':', linewidth=2, label=f"Overall CG ({cg_x_total:.3f} m)")
    ax.axvline(wing_ac[0], color='green', linestyle=':', linewidth=2, label=f"Wing AC ({wing_ac[0]:.3f} m)")
    ax.axvline(x_np, color='purple', linestyle=':', linewidth=2, label=f"Neutral Point ({x_np:.3f} m)")

    ax.set_title("Longitudinal Mass & Aero Layout", fontsize=16)
    ax.set_xlabel("X-Position (m)")
    ax.set_yticks([])
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    out = save_directory / f"{config_name}_mass_distribution_{timestamp}.png"
    fig.savefig(out, dpi=dpi)
    print(f"Saved mass distribution plot to: {out.resolve()}")
    plt.close(fig)

# === Numeric re-evaluator used for diagnostics (no MX/CasADi gradients) ===
def _evaluate_terms_numeric(
    vals,
    *,
    airfoils, E_OSWALD, RHO, CRUISE_TO_VSTALL, CLMAX_MARGIN, WING_CLMAX_LOOKUP, nu=1.5e-5
):
    """
    Pure-numeric re-eval of LD, Vsink, Vstall, CDi using a minimal reconstruction
    of geometry and aero at *numeric* values. Avoids CasADi symbolic gradients.
    """
    import aerosandbox as asb
    import aerosandbox.numpy as np
    import aerosandbox.tools.units as u

    # ---- Unpack (numbers only)
    wing_airfoil_name = vals["wing_airfoil_name"]
    vtail_airfoil_name = vals["vtail_airfoil_name"]

    AF_wing  = airfoils[wing_airfoil_name]
    AF_vtail = airfoils[vtail_airfoil_name]

    # Geometry (numbers)
    wing_span         = vals["wing_span"]
    wing_root_chord   = vals["wing_root_chord"]
    wing_taper_ratio  = vals["wing_taper_ratio"]
    wing_dihedral     = 0.0
    wing_center_span_fraction = 0.5

    winglet_height_fraction = vals["winglet_height_fraction"]
    winglet_cant_angle      = vals["winglet_cant_angle"]
    winglet_sweep_angle     = vals["winglet_sweep_angle"]
    winglet_taper_ratio     = vals["winglet_taper_ratio"]

    vtail_span        = vals["vtail_span"]
    vtail_root_chord  = vals["vtail_root_chord"]
    vtail_taper_ratio = vals["vtail_taper_ratio"]
    vtail_dihedral    = vals["vtail_dihedral"]
    vtail_le_sweep    = vals["vtail_le_sweep"]
    i_tail_deg = vals.get("i_tail_deg", 0.0)

    boom_nose_x       = vals["boom_nose_x"]
    x_tail            = vals["x_tail"]
    cylinder_center_x = vals["cylinder_center_x"]

    engine_span_fraction = vals["engine_span_fraction"]
    wing_motor_x_frac    = vals["wing_motor_x_frac"]  # not strictly needed here, but harmless

    # Fixed pod/boom parameters (match your script)
    CYL_DIAM = 3 * u.inch
    CYL_LEN  = 8 * u.inch
    boom_radius = 0.003

    # --- Build wing (numeric)
    wing_tip_chord = wing_root_chord * wing_taper_ratio
    wing_root_xsec = asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=AF_wing)
    wing_break_xsec = asb.WingXSec(
        xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
        chord=wing_root_chord, airfoil=AF_wing
    )
    wing_tip_xsec = asb.WingXSec(
        xyz_le=[-0.25 * wing_tip_chord, wing_span / 2,
                wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral)],
        chord=wing_tip_chord, airfoil=AF_wing
    )
    # Winglet
    winglet_chord = wing_tip_chord * winglet_taper_ratio
    winglet_span_projection = (wing_span / 2 * winglet_height_fraction)
    delta_y_winglet = winglet_span_projection * np.cosd(winglet_cant_angle) + 1e-6
    delta_z_winglet = winglet_span_projection * np.sind(winglet_cant_angle)
    delta_x_winglet = winglet_span_projection * np.tand(winglet_sweep_angle)
    winglet_tip_xsec = asb.WingXSec(
        xyz_le=[wing_tip_xsec.xyz_le[0] + delta_x_winglet,
                wing_tip_xsec.xyz_le[1] + delta_y_winglet,
                wing_tip_xsec.xyz_le[2] + delta_z_winglet],
        chord=winglet_chord, airfoil=AF_wing
    )
    wing = asb.Wing(
        name="Main Wing (numeric)", symmetric=True,
        xsecs=[wing_root_xsec, wing_break_xsec, wing_tip_xsec, winglet_tip_xsec]
    )

    # --- Build V-tail (numeric)
    vtail_tip_chord = vtail_root_chord * vtail_taper_ratio
    vtail_tip_x_le  = vtail_span / 2 * np.tand(vtail_le_sweep)
    vtail = asb.Wing(
        name="V-Tail (numeric)", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=vtail_root_chord, airfoil=AF_vtail, twist=i_tail_deg),
            asb.WingXSec(
                xyz_le=[vtail_tip_x_le, vtail_span / 2, (vtail_span / 2) * np.sind(vtail_dihedral)],
                chord=vtail_tip_chord, airfoil=AF_vtail, twist=i_tail_deg
            ),
        ]
    ).translate([x_tail, 0.0, 0.0])

    # --- Pod + boom (numeric)
    cylinder_start_x = cylinder_center_x - CYL_LEN / 2
    cylinder_end_x   = cylinder_center_x + CYL_LEN / 2
    cylinder_body = asb.Fuselage(
        name="Cylinder Body (numeric)",
        xsecs=[
            asb.FuselageXSec(xyz_c=[cylinder_start_x, 0, 0], radius=CYL_DIAM / 2),
            asb.FuselageXSec(xyz_c=[cylinder_end_x,   0, 0], radius=CYL_DIAM / 2),
        ]
    )
    tail_boom = asb.Fuselage(
        name="Tail Boom (numeric)",
        xsecs=[
            asb.FuselageXSec(xyz_c=[boom_nose_x, 0, 0], radius=boom_radius),
            asb.FuselageXSec(xyz_c=[x_tail,      0, 0], radius=boom_radius),
        ]
    )

    airplane = asb.Airplane(
        name="Numeric Clone",
        wings=[wing, vtail],
        fuselages=[cylinder_body, tail_boom]
    )

    # ---- Mass / speeds
    mass_total = vals["mass_total"]
    xcg_total  = vals["xcg_total"]
    mass_props_TOGW = asb.MassProperties(mass=mass_total, x_cg=xcg_total, y_cg=0.0, z_cg=0.0)

    # CLmax estimate (2D) with safe fallback
    try:
        alphas = np.linspace(-5, 24, 60)
        CLs = wing.xsecs[0].airfoil.CL_function(alpha=alphas)
        clmax_2d = float(np.max(CLs))
    except Exception:
        clmax_2d = float(WING_CLMAX_LOOKUP.get(wing_airfoil_name, 1.2))
    CLmax_est = CLMAX_MARGIN * clmax_2d

    W = mass_total * 9.81
    S = wing.area()
    Vstall  = np.sqrt(2 * W / (RHO * (S * (CLmax_est + 1e-9))))
    Vcruise = CRUISE_TO_VSTALL * Vstall

    # ---- Aerodynamics at cruise: find alpha that trims Cm≈0 (tiny bracketed search)
    def aero_at_alpha(alpha_deg: float):
        ab = asb.AeroBuildup(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=Vcruise, alpha=alpha_deg),
            xyz_ref=mass_props_TOGW.xyz_cg
        ).run()
        return ab

    a_lo, a_hi = -2.0, 8.0
    for _ in range(12):
        a_mid = 0.5 * (a_lo + a_hi)
        Cm_lo = aero_at_alpha(a_lo)["Cm"]
        Cm_mid = aero_at_alpha(a_mid)["Cm"]
        if Cm_lo * Cm_mid <= 0:
            a_hi = a_mid
        else:
            a_lo = a_mid
    a_trim = 0.5 * (a_lo + a_hi)
    aero = aero_at_alpha(a_trim)

    CL = aero["CL"]; CD = aero["CD"]; L = aero["L"]; D = aero["D"]
    LD = L / (D + 1e-12)
    AR = (wing.span() ** 2) / (S + 1e-12)
    CDi = (CL ** 2) / (np.pi * E_OSWALD * (AR + 1e-9))

    # Min-sink surrogate at provided alpha_ms (or trim+2° fallback)
    alpha_ms = vals.get("alpha_ms", a_trim + 2.0)
    ab_ms = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=Vcruise, alpha=alpha_ms),
        xyz_ref=mass_props_TOGW.xyz_cg
    ).run()
    CL_ms = ab_ms["CL"]; CD_ms = ab_ms["CD"]
    Vsink = np.sqrt(2 * W / (RHO * (S + 1e-12))) * (CD_ms / (CL_ms ** 1.5 + 1e-9))

    return dict(LD=float(LD), Vsink=float(Vsink), Vstall=float(Vstall), CDi=float(CDi))

def _avl_composite_report_save(config_name, airplane, op_point, mass_props, mass_props_TOGW, sol, dpi=200):
    print_title("Generating AVL-Based Report")
    try:
        # ---- helpers ----
        def _fmt_scalar(v):
            import numpy as _np
            try:
                v = _np.asarray(v)
                if v.size == 1:
                    v = float(v.reshape(()))  # 0-D array to Python float
                else:
                    return str(v)
                return f"{v: .5g}"
            except Exception:
                try:
                    return f"{float(v): .5g}"
                except Exception:
                    return str(v)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deg2rad = np.pi / 180.0

        # --- (1) Polar sweep via per-alpha calls (AVL) ---
        alpha_range = np.linspace(-15, 25, 41)
        CLs, CDs, Cms = [], [], []
        for a in alpha_range:
            res = asb.AVL(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=float(op_point.velocity), alpha=float(a)),
                xyz_ref=mass_props_TOGW.xyz_cg
            ).run()
            CLs.append(res["CL"]); CDs.append(res["CD"]); Cms.append(res["Cm"])
        aa = np.array(alpha_range)
        aero_polars_avl = {"alpha": aa, "CL": np.array(CLs), "CD": np.array(CDs), "Cm": np.array(Cms)}

        # --- (2) Single-point AVL run at cruise (for whatever keys AVL returns) ---
        op_cruise_num = asb.OperatingPoint(
            velocity=float(op_point.velocity),
            alpha=float(op_point.alpha),
            beta=float(getattr(op_point, "beta", 0.0)),
            p=float(getattr(op_point, "p", 0.0)),
            q=float(getattr(op_point, "q", 0.0)),
            r=float(getattr(op_point, "r", 0.0)),
        )
        try:
            avl_stab_results = asb.AVL(
                airplane=airplane,
                op_point=op_cruise_num,
                xyz_ref=mass_props_TOGW.xyz_cg
            ).run()
        except Exception as e:
            print(f"[AVL] single-point run failed (continuing with FD): {e}")
            avl_stab_results = {}

        # --- (3) Near-trim finite-difference derivatives around cruise (AVL & AeroBuildup) ---
        a0 = float(op_point.alpha)
        da = 1  # deg
        c_ref = float(airplane.wings[0].mean_aerodynamic_chord())
        x_ref = float(mass_props_TOGW.x_cg)  # xyz_ref == CG

        def avl_at(alpha_deg: float):
            return asb.AVL(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=float(op_point.velocity), alpha=float(alpha_deg)),
                xyz_ref=mass_props_TOGW.xyz_cg
            ).run()

        res_m = avl_at(a0 - da)
        res_0 = avl_at(a0)
        res_p = avl_at(a0 + da)

        CL_alpha_avl = (res_p["CL"] - res_m["CL"]) / (2 * da) / deg2rad   # per rad
        Cm_alpha_avl = (res_p["Cm"] - res_m["Cm"]) / (2 * da) / deg2rad   # per rad
        x_np_avl = x_ref - (Cm_alpha_avl / (CL_alpha_avl + 1e-12)) * c_ref
        SM_avl   = (x_np_avl - x_ref) / c_ref

        # AeroBuildup (AB) finite-difference comparison (same a0, da)
        def AB_at(alpha_deg: float):
            return asb.AeroBuildup(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=float(op_point.velocity), alpha=float(alpha_deg)),
                xyz_ref=mass_props_TOGW.xyz_cg
            ).run()

        ab_m = AB_at(a0 - da)
        ab_0 = AB_at(a0)
        ab_p = AB_at(a0 + da)

        CL_alpha_ab = (ab_p["CL"] - ab_m["CL"]) / (2 * da) / deg2rad
        Cm_alpha_ab = (ab_p["Cm"] - ab_m["Cm"]) / (2 * da) / deg2rad
        SM_ab       = -(Cm_alpha_ab / (CL_alpha_ab + 1e-12))  # xyz_ref = CG

        print_title("Near-trim linearization (per rad)")
        print(f"AVL: CL_alpha={float(CL_alpha_avl):8.3f}, Cm_alpha={float(Cm_alpha_avl):9.4f}, SM={float(SM_avl)*100:6.2f}%")
        print(f" AB: CL_alpha={float(CL_alpha_ab):8.3f}, Cm_alpha={float(Cm_alpha_ab):9.4f}, SM={float(SM_ab)*100:6.2f}%")

        # --- (4) Figure (AVL polars, SM using FD x_np_avl, and linearized Cm tangent) ---
        fig = plt.figure(figsize=(24, 18), dpi=dpi, constrained_layout=True)
        fig.suptitle(f"{config_name.replace('_', ' ')}: L/D and Stability Report (using AVL)", fontsize=24, y=0.98)
        gs = fig.add_gridspec(3, 3)

        # Three-view
        ax_plane = fig.add_subplot(gs[0, 0:2])
        try:
            tv_img = _three_view_to_array(airplane, dpi=max(200, dpi))
            ax_plane.imshow(tv_img)
            ax_plane.set_title("Airplane (Three-View Rendering)", fontsize=16)
            ax_plane.axis("off")
        except Exception as e:
            ax_plane.text(0.5, 0.5, f"three_view render failed:\n{e}", ha="center", va="center")
            ax_plane.axis("off")

        # Mass pie
        ax_pie = fig.add_subplot(gs[0, 2])
        mass_props_to_plot = {k: v for k, v in mass_props.items() if float(v.mass) > 1e-9}
        plt.sca(ax_pie)
        p.pie(
            values=[v.mass for v in mass_props_to_plot.values()],
            names=[k.replace("_", " ").title() for k in mass_props_to_plot.keys()],
            center_text=(f"Mass Budget\nTOGW: {mass_props_TOGW.mass * 1e3:.2f} g"),
            startangle=110, arm_length=28, arm_radius=18, y_max_labels=1.10
        )
        ax_pie.set_title("Mass Budget", fontsize=16, pad=16)

        # CL, CD, L/D
        ax_cl = fig.add_subplot(gs[1, 0]); ax_cd = fig.add_subplot(gs[1, 1]); ax_ld = fig.add_subplot(gs[1, 2])
        CL = aero_polars_avl["CL"]; CD = aero_polars_avl["CD"]
        ax_cl.plot(aa, CL); ax_cd.plot(aa, CD); ax_ld.plot(aa, CL / (CD + 1e-12))
        ax_cl.set_xlabel("α [deg]"); ax_cl.set_ylabel("CL"); ax_cl.set_title("CL vs α"); ax_cl.grid(True)
        ax_cd.set_xlabel("α [deg]"); ax_cd.set_ylabel("CD"); ax_cd.set_title("CD vs α"); ax_cd.grid(True)
        ax_ld.set_xlabel("α [deg]"); ax_ld.set_ylabel("L/D"); ax_ld.set_title("L/D vs α"); ax_ld.grid(True)

        # Static margin (use FD x_np) & Cm
        ax_sm = fig.add_subplot(gs[2, 0]); ax_cm = fig.add_subplot(gs[2, 1])
        mac = c_ref
        x_np = float(x_np_avl)  # robust FD neutral point
        sm = (x_np - x_ref) / mac
        x_cg_range = np.linspace(x_np - 0.5 * mac, x_np + 0.5 * mac, 200)
        sm_range = (x_np - x_cg_range) / mac
        ax_sm.plot(x_cg_range, sm_range * 100)
        ax_sm.axvline(x_ref, color='r', linestyle='--', label=f"SM = {sm*100:.1f}%")
        ax_sm.axhline(0, color='k', lw=0.8)
        ax_sm.set_xlabel("CG [m]"); ax_sm.set_ylabel("Static Margin [% MAC]")
        ax_sm.set_title("Static Margin vs CG"); ax_sm.legend(); ax_sm.grid(True)

        ax_cm.plot(aa, aero_polars_avl["Cm"], label="AVL Cm(α)")
        Cm0 = float(res_0["Cm"])
        ax_cm.plot(aa, Cm0 + float(Cm_alpha_avl) * (aa - a0) * deg2rad, "--", label="AVL linear")
        ax_cm.axhline(0, color='k', lw=0.8)
        ax_cm.set_xlabel("α [deg]"); ax_cm.set_ylabel("Cm"); ax_cm.set_title("Cm vs α")
        ax_cm.grid(True); ax_cm.legend()

        out = save_directory / f"{config_name}_AVL_report_{timestamp}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved AVL report to: {out.resolve()}")
        plt.close(fig)

        # --- (5) Print whatever derivatives AVL returned (robust formatting) ---
        if avl_stab_results:
            print_title("AVL stability derivatives @ cruise (from single-point run)")
            for k in ["x_np", "CLa", "CL_a", "Cma", "Cmq", "Cl_p", "Cl_beta", "Cn_beta", "Cn_r"]:
                if k in avl_stab_results:
                    print(f"{k.rjust(10)} = {_fmt_scalar(avl_stab_results[k])}")

    except Exception as e:
        print(f"\n--- [WARNING] Could not generate AVL-based report! ---")
        print(f"  Error: {e}")
        traceback.print_exc()

def estimate_clmax_2d_and_alpha_stall(airfoil_obj, fallback_name: str):
    try:
        alphas = np.linspace(-5, 24, 60)
        try:
            CLs = airfoil_obj.CL_function(alpha=alphas)
        except Exception:
            CLs = np.array([airfoil_obj.get_aero_from_alpha(alpha=a)["CL"] for a in alphas])
        i_max = int(np.argmax(CLs))
        clmax_2d = float(CLs[i_max]); alpha_stall_2d = float(alphas[i_max])
        return clmax_2d, alpha_stall_2d
    except Exception:
        return float(WING_CLMAX_LOOKUP.get(fallback_name, 1.2)), None

def report_solution(sol, config_name, airplane, op_point, mass_props, mass_props_TOGW, LD_expr, aero, opti_vars,
                    Vsink=None, Vstall=None, CDi=None,
                    boom_nose_x=None, x_tail=None, wing=None):
    s = lambda x: x
    def fmt(x):
        try:
            return f"{s(x):.6g}"
        except:
            return "N/A"
    def fmt_stab(*keys):
        for k in keys:
            v = aero.get(k, None)
            if v is not None:
                try:
                    return f"{v:.6g}"
                except:
                    return "N/A"
        return "N/A"

    print_title(f"{config_name} Outputs")
    output_data = {
        "TOGW": f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "Wing Loading": f"{fmt(mass_props_TOGW.mass * 9.81 / airplane.wings[0].area())} N/m^2",
        "L/D Cruise": fmt(LD_expr),
        "Cruise Airspeed": f"{fmt(op_point.velocity)} m/s",
        "Cruise AoA": f"{fmt(op_point.alpha)} deg",
        "Wing Span": f"{fmt(airplane.wings[0].span())} m",
        "Wing MAC": f"{fmt(airplane.wings[0].mean_aerodynamic_chord())} m",
        "Center of Gravity (x,y,z)": f"({s(mass_props_TOGW.xyz_cg[0]):.4f}, {s(mass_props_TOGW.xyz_cg[1]):.4f}, {s(mass_props_TOGW.xyz_cg[2]):.4f}) m",
        "Static Margin": f"{fmt((aero['x_np'] - mass_props_TOGW.x_cg) / s(airplane.wings[0].mean_aerodynamic_chord()))}",
        "Cma": fmt_stab("Cma"),
        "Cn_beta/Cnb": fmt_stab("Cn_beta", "Cnb"),
        "Cl_p/Clp": fmt_stab("Cl_p", "Clp"),
        "Cmq": fmt_stab("Cmq"),
        "Cn_r/Cnr": fmt_stab("Cn_r", "Cnr"),
        "Cl_beta/Clb": fmt_stab("Cl_beta", "Clb"),
    }
    if Vsink is not None:
        output_data["Min-Sink Rate (est)"] = f"{fmt(Vsink)} m/s"
    if Vstall is not None:
        output_data["Stall Speed (est)"] = f"{fmt(Vstall)} m/s"

    for k, v in output_data.items():
        print(f"{k.rjust(25)} = {v}")

    print_title("Mass Budget")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {s(v.mass) * 1e3:.2f} g ({s(v.mass) / u.oz:.2f} oz)")

    print_title(f"{config_name} Optimized Design Variables")
    for name, var_val in opti_vars.items():
        try:
            print(f"  {name.ljust(35)}: {var_val:.6g}")
        except:
            pass

    if all(v is not None for v in [LD_expr, Vsink, Vstall, CDi]):
        print_title("Objective Function Breakdown")
        ld_score     = LD_WEIGHT * LD_expr
        vsink_score  = -SINK_WEIGHT * Vsink
        vstall_score = -STALL_WEIGHT * Vstall
        cdi_score    = -INDUCED_PENALTY_WEIGHT * CDi
        mass_score = float(-MASS_WEIGHT * mass_props_TOGW.mass)
        total_J = ld_score + vsink_score + vstall_score + cdi_score
        print(f"{'Component':<28} | {'Score Contribution'}")
        print("-" * 50)
        print(f"{'L/D Contribution':<28} | {ld_score: >20.4f}")
        print(f"{'Sink Rate Penalty':<28} | {vsink_score: >20.4f}")
        print(f"{'Stall Speed Penalty':<28} | {vstall_score: >20.4f}")
        print(f"{'Induced Drag Penalty':<28} | {cdi_score: >20.4f}")
        print(f"{'Mass Penalty':<28} | {mass_score: >20.4f}")
        print("-" * 50)
        print(f"{'Total Objective J':<28} | {total_J: >20.4f}")

    # ----- Geometry Exports -----
    print_title("Geometry Exports")
    try:
        # Make a run-specific timestamp so we don't overwrite prior outputs
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build file paths in your existing save directory
        avl_path        = save_directory / f"{config_name}_{ts_now}.avl"
        vsp_script_path = save_directory / f"{config_name}_{ts_now}.vspscript"
        xflr5_path      = save_directory / f"{config_name}_{ts_now}.xml"

        # Export using AeroSandbox helpers (airplane here is already numeric)
        airplane.export_AVL(filename=str(avl_path))
        print(f"  Exported AVL geometry to: {avl_path.resolve()}")

        airplane.export_OpenVSP_vspscript(filename=str(vsp_script_path))
        print(f"  Exported OpenVSP script to: {vsp_script_path.resolve()}")

        airplane.export_XFLR5_xml(filename=str(xflr5_path))
        print(f"  Exported XFLR5 (xml) geometry to: {xflr5_path.resolve()}")

    except Exception as e:
        print(f"\n--- [WARNING] Geometry export failed! ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("-------------------------------------------\n")


    if MAKE_PLOTS:
        # PNG 1: AeroBuildup-based report
        _composite_report_save(config_name, airplane, op_point, mass_props, mass_props_TOGW, aero, sol, dpi=200)

        # PNG 2: AVL-based report (independent check)
        _avl_composite_report_save(config_name, airplane, op_point, mass_props, mass_props_TOGW, sol, dpi=200)

        # PNG 3: Mass distribution
        if boom_nose_x is not None and x_tail is not None and wing is not None:
            _plot_mass_distribution(
                config_name=config_name,
                mass_props=mass_props,
                mass_props_TOGW=mass_props_TOGW,
                boom_nose_x=boom_nose_x,
                x_tail=x_tail,
                wing_ac=wing.aerodynamic_center(),
                x_np=aero['x_np']
            )


# ----------------- Core solve per airfoil combo -----------------
def build_and_solve(wing_airfoil_name: str, vtail_airfoil_name: str):
    try:
        opti = asb.Opti()

        # ----- Flight condition angles
        alpha_cruise = opti.variable(init_guess=0.0, lower_bound=-10, upper_bound=20, scale=1)
        alpha_ms     = opti.variable(init_guess=4.0, lower_bound=-10, upper_bound=20, scale=1)

        # ----- Geometry variables (Cylinder + Boom + Tail)
        boom_nose_x = opti.variable(init_guess=-0.3, upper_bound=-0.1, scale=0.1)
        x_tail      = opti.variable(init_guess=0.5, lower_bound=0.4, scale=1)  # teammate-style LB = 0.3
        boom_radius = 0.003

        CYL_DIAM = 3 * u.inch
        CYL_LEN  = 8 * u.inch
        cylinder_center_x = opti.variable(init_guess=-0.2, scale=0.1)
        cylinder_start_x  = cylinder_center_x - CYL_LEN / 2
        cylinder_end_x    = cylinder_center_x + CYL_LEN / 2

        # Wing geometry
        wing_span = opti.variable(init_guess=1.5, lower_bound=0.4, upper_bound=2, scale=1)
        wing_root_chord = opti.variable(init_guess=0.2, lower_bound=0.102, upper_bound=0.5, scale=0.1)
        wing_taper_ratio = opti.variable(init_guess=0.7, lower_bound=0.2, upper_bound=1.0, scale=1)
        wing_dihedral = 0
        wing_center_span_fraction = 0.5

        # Winglet
        winglet_height_fraction = opti.variable(init_guess=0.1, lower_bound=0.00, upper_bound=0.15, scale=0.1)
        winglet_cant_angle = opti.variable(init_guess=30, lower_bound=25, upper_bound=90, scale=100)
        winglet_sweep_angle = opti.variable(init_guess=15, lower_bound=0, upper_bound=45, scale=10)
        winglet_taper_ratio = opti.variable(init_guess=0.6, lower_bound=0.1, upper_bound=1.0, scale=1)

        # V-Tail geometry
        vtail_span = opti.variable(init_guess=0.25, lower_bound=0.05, upper_bound=0.8, scale=0.1)
        vtail_root_chord = opti.variable(init_guess=0.2, lower_bound=0.13, upper_bound=0.3, scale=0.1)
        vtail_taper_ratio = opti.variable(init_guess=0.7, lower_bound=0.5, upper_bound=1.0, scale=1)
        vtail_dihedral = opti.variable(init_guess=35, lower_bound=15, upper_bound=43, scale=10)
        vtail_le_sweep = opti.variable(init_guess=10, lower_bound=0, upper_bound=25, scale=10)
        i_tail_deg = opti.variable(init_guess=-2.0, lower_bound=-5.0, upper_bound=0.0)

        # Engine placement
        engine_span_fraction = opti.variable(init_guess=0.3, lower_bound=0.1, upper_bound=0.9, scale=1)
        wing_motor_x_frac = opti.variable(init_guess=0.30, lower_bound=0.00, upper_bound=0.80, scale=0.1)

        # ----- Build components with selected airfoils
        AF_wing = airfoils[wing_airfoil_name]
        AF_vtail = airfoils[vtail_airfoil_name]

        wing_tip_chord = wing_root_chord * wing_taper_ratio
        wing_root_xsec = asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, 0, 0], chord=wing_root_chord, airfoil=AF_wing)
        wing_break_xsec = asb.WingXSec(xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
                                       chord=wing_root_chord, airfoil=AF_wing)
        wing_tip_xsec = asb.WingXSec(
            xyz_le=[-0.25 * wing_tip_chord, wing_span / 2, wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral)],
            chord=wing_tip_chord, airfoil=AF_wing
        )
        # Winglet geometry
        winglet_chord = wing_tip_chord * winglet_taper_ratio
        winglet_span_projection = (wing_span / 2 * winglet_height_fraction)
        delta_y_winglet = winglet_span_projection * np.cosd(winglet_cant_angle) + 1e-6  # avoid zero-span
        delta_z_winglet = winglet_span_projection * np.sind(winglet_cant_angle)
        delta_x_winglet = winglet_span_projection * np.tand(winglet_sweep_angle)
        winglet_tip_le_x = wing_tip_xsec.xyz_le[0] + delta_x_winglet
        winglet_tip_le_y = wing_tip_xsec.xyz_le[1] + delta_y_winglet
        winglet_tip_le_z = wing_tip_xsec.xyz_le[2] + delta_z_winglet
        winglet_tip_xsec = asb.WingXSec(
            xyz_le=[winglet_tip_le_x, winglet_tip_le_y, winglet_tip_le_z],
            chord=winglet_chord,
            airfoil=AF_wing
        )
        wing = asb.Wing(name=f"Main Wing ({wing_airfoil_name})", symmetric=True,
                        xsecs=[wing_root_xsec, wing_break_xsec, wing_tip_xsec, winglet_tip_xsec])

        # V-tail
        vtail_tip_chord = vtail_root_chord * vtail_taper_ratio
        vtail_tip_x_le = vtail_span / 2 * np.tand(vtail_le_sweep)
        vtail = asb.Wing(
            name=f"V-Tail ({vtail_airfoil_name})", symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=vtail_root_chord, airfoil=AF_vtail, twist=i_tail_deg),
                asb.WingXSec(xyz_le=[vtail_tip_x_le, vtail_span / 2, (vtail_span / 2) * np.sind(vtail_dihedral)],
                             chord=vtail_tip_chord, airfoil=AF_vtail, twist=i_tail_deg),
            ]
        ).translate([x_tail, 0.0, 0.0])

        # Tail aspect ratio (symbolic)
        AR_tail = (vtail.span() ** 2) / (vtail.area() + 1e-9)
        opti.subject_to(AR_tail >= 7.0)  # after vtail is built

        vt_semi = vtail.span() / 2
        delta_x_te_tail = vt_semi * np.tand(vtail_le_sweep) + vtail_taper_ratio * vtail_root_chord - vtail_root_chord
        opti.subject_to(delta_x_te_tail >= - vt_semi * np.tand(20))

        # Cylinder body and tail boom
        cylinder_body = asb.Fuselage(
            name="Cylinder Body",
            xsecs=[
                asb.FuselageXSec(xyz_c=[cylinder_start_x, 0, 0], radius=CYL_DIAM / 2),
                asb.FuselageXSec(xyz_c=[cylinder_end_x,   0, 0], radius=CYL_DIAM / 2),
            ]
        )
        tail_boom = asb.Fuselage(
            name="Tail Boom",
            xsecs=[
                asb.FuselageXSec(xyz_c=[boom_nose_x, 0, 0], radius=boom_radius),
                asb.FuselageXSec(xyz_c=[x_tail,      0, 0], radius=boom_radius),
            ]
        )

        airplane = asb.Airplane(
            name="Feather VTail (Pod+Boom)",
            wings=[wing, vtail],
            fuselages=[cylinder_body, tail_boom]
        )

        # ----- Mass properties
        mass_props = {}
        FUSELAGE_DENS = 100  # kg/m^3

        mass_props['cylinder_body'] = asb.MassProperties(
            mass=cylinder_body.volume() * FUSELAGE_DENS,
            x_cg=cylinder_center_x
        )

        wing_area = wing.area()
        m_wing_skin = WING_SKIN_AREAL_DENS * wing_area

        mass_props["wing_skin"] = asb.MassProperties(mass=m_wing_skin,
                                                     x_cg=wing.aerodynamic_center()[0])

        vtail_area = vtail.area()
        m_vtail_skin = VTAIL_SKIN_AREAL_DENS * vtail_area

        mass_props["vtail_skin"] = asb.MassProperties(mass=m_vtail_skin,
                                                      x_cg=vtail.aerodynamic_center()[0])

        # --- Spar mass models based on your actual parts (linear-by-length)

        # Material density (can be updated if you measure a piece)
        CF_DENSITY = 1600.0  # kg/m^3

        # Section properties
        A_tube_4x2mm = np.pi * ((2e-3) ** 2 - (1e-3) ** 2)  # m^2
        A_rod_2mm = np.pi * (1e-3) ** 2  # m^2

        lambda_tube_4x2mm = CF_DENSITY * A_tube_4x2mm  # kg/m ≈ 0.01508
        lambda_rod_2mm = CF_DENSITY * A_rod_2mm  # kg/m ≈ 0.00503

        # ---------------- WING SPARS ----------------
        # Geometry
        b_full = wing.span()  # full span (m)
        b_half = b_full / 2  # semi-span (m)

        # Per your build: 2 spars per side (so 4 total along the full wing)
        n_spars_per_side = 2
        n_sides = 2
        n_wing_spars_total = n_spars_per_side * n_sides  # 4

        # Outer section length on each side (2 mm rods)
        L_outer_per_side = 0.25  # meters, from tip inward
        L_outer_per_side = np.minimum(L_outer_per_side, b_half)  # cap if semi-span is shorter
        L_inner_per_side = np.maximum(b_half - L_outer_per_side, 0.0)

        # Mass per spar on one side
        m_per_spar_one_side = (
                lambda_tube_4x2mm * L_inner_per_side +
                lambda_rod_2mm * L_outer_per_side
        )

        # Total mass across both sides and both spars
        m_wing_spar = n_wing_spars_total * m_per_spar_one_side

        mass_props["wing_spar"] = asb.MassProperties(
            mass=m_wing_spar,
            x_cg=wing.aerodynamic_center()[0]
        )

        # ---------------- V-TAIL SPARS ----------------
        # Per your build: 2 rods per side (so 4 total), all 2 mm solid rods
        vtail_b_full = vtail.span()
        vtail_b_half = vtail_b_full / 2

        n_tail_rods_per_side = 2
        n_tail_sides = 2
        n_tail_rods_total = n_tail_rods_per_side * n_tail_sides  # 4

        # Each rod runs along the semi-span (simple assumption)
        m_vtail_spar = n_tail_rods_total * (lambda_rod_2mm * vtail_b_half)

        mass_props["vtail_spar"] = asb.MassProperties(
            mass=m_vtail_spar,
            x_cg=vtail.aerodynamic_center()[0]
        )

        # ---------- Boom mass (linear density) ----------
        lambda_boom = 0.0303 / 1.19  # kg/m
        boom_length = np.maximum(x_tail - boom_nose_x, 0.0)
        m_boom = lambda_boom * boom_length

        mass_props['tail_boom'] = asb.MassProperties(
            mass=m_boom,
            x_cg=(boom_nose_x + x_tail) / 2
        )

        # --- Point masses
        x_cg_base_electronics      = opti.variable(init_guess=0.0, lower_bound=cylinder_start_x, upper_bound=cylinder_start_x+0.33*CYL_LEN, scale=0.1)
        x_cg_vtol_electronics      = opti.variable(init_guess=0.0, lower_bound=cylinder_start_x, upper_bound=cylinder_start_x+0.33*CYL_LEN, scale=0.1)
        x_cg_non_vtol_electronics  = opti.variable(init_guess=0.0, lower_bound=cylinder_start_x, upper_bound=cylinder_start_x+0.33*CYL_LEN, scale=0.1)
        x_cg_cv_electronics        = opti.variable(init_guess=0.0, lower_bound=cylinder_start_x, upper_bound=cylinder_start_x+0.33*CYL_LEN, scale=0.1)

        # Payload bounds like teammate: between wing AC and tail
        x_cg_payload = opti.variable(
            init_guess=0.0,
            lower_bound=wing.aerodynamic_center()[0],
            upper_bound=x_tail,
            scale=0.1
        )

        # Wing motors (two) and landing gear
        M_WING_MOTOR_KG = 0.039  # each
        M_LANDING_GEAR_KG = 0.050

        engine_y_pos = engine_span_fraction * wing.span() / 2
        wing_xsecs_y_le = np.stack([xsec.xyz_le[1] for xsec in wing.xsecs])
        wing_xsecs_x_le = np.stack([xsec.xyz_le[0] for xsec in wing.xsecs])
        wing_xsecs_z_le = np.stack([xsec.xyz_le[2] for xsec in wing.xsecs])

        x_le_at_motor = np.interp(engine_y_pos, wing_xsecs_y_le, wing_xsecs_x_le)
        z_le_at_motor = np.interp(engine_y_pos, wing_xsecs_y_le, wing_xsecs_z_le)

        x_cg_motors = x_le_at_motor + wing_motor_x_frac * wing_root_chord
        z_cg_motors = z_le_at_motor

        mass_props["left_wing_motor"] = asb.MassProperties(
            mass=M_WING_MOTOR_KG,
            x_cg=x_cg_motors,
            y_cg=-engine_y_pos,
            z_cg=z_cg_motors
        )
        mass_props["right_wing_motor"] = asb.MassProperties(
            mass=M_WING_MOTOR_KG,
            x_cg=x_cg_motors,
            y_cg=engine_y_pos,
            z_cg=z_cg_motors
        )

        x_cg_gear = opti.variable(init_guess=-0.18, lower_bound=cylinder_start_x, upper_bound=cylinder_end_x, scale=0.1)
        mass_props["landing_gear"] = asb.MassProperties(mass=M_LANDING_GEAR_KG, x_cg=x_cg_gear)

        # Rear engine between wing AC and tail
        x_cg_rear_engine = opti.variable(init_guess=0.3, lower_bound=wing.aerodynamic_center()[0], upper_bound=x_tail, scale=0.1)

        # Payload mass fixed to 1.0 lbm
        payload_mass = 1.0 * u.lbm

        mass_props["base_electronics"]     = asb.MassProperties(mass=95e-3,  x_cg=x_cg_base_electronics)
        mass_props["vtol_electronics"]     = asb.MassProperties(mass=141e-3, x_cg=x_cg_vtol_electronics)
        mass_props["non_vtol_electronics"] = asb.MassProperties(mass=120e-3, x_cg=x_cg_non_vtol_electronics)
        mass_props["cv_electronics"]       = asb.MassProperties(mass=17e-3,  x_cg=x_cg_cv_electronics)
        mass_props["rear_engine"]          = asb.MassProperties(mass=31e-3,  x_cg=x_cg_rear_engine)
        mass_props["payload"]              = asb.MassProperties(mass=payload_mass, x_cg=x_cg_payload)

        mass_props_TOGW = sum(mass_props.values())

        # ---- Payload ≈ total CG (3 cm tolerance, teammate-style)
        opti.subject_to(np.abs(x_cg_payload - mass_props_TOGW.x_cg) <= 0.03)

        # ---- Hard-fix TOGW upper bound (keep your original)
        opti.subject_to(mass_props_TOGW.mass <= 3.5 * u.lbm)

        # ----- Compute Vstall and Vcruise using fixed mass
        W = mass_props_TOGW.mass * 9.81
        S = wing.area()
        clmax_2d, _ = estimate_clmax_2d_and_alpha_stall(AF_wing, wing_airfoil_name)
        CLmax_est = CLMAX_MARGIN * clmax_2d
        Vstall  = np.sqrt(2 * W / (RHO * (S * (CLmax_est + 1e-9))))
        Vcruise = CRUISE_TO_VSTALL * Vstall

        # ----- Aerodynamics @ cruise
        op_point = asb.OperatingPoint(velocity=Vcruise, alpha=alpha_cruise)
        ab   = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=mass_props_TOGW.xyz_cg)
        aero = ab.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)

        # Lift/trim
        LD_cruise = opti.variable(init_guess=15, lower_bound=5)
        LD = aero["L"] / (aero["D"] + 1e-9)
        opti.subject_to(LD_cruise == LD)
        opti.subject_to([aero["L"] >= W, aero["Cm"] == 0])

        # Aspect ratio ≥ 4
        AR = (wing.span() ** 2) / (wing.area() + 1e-9)
        opti.subject_to(AR >= 4.0)

        # Static margin band
        static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / (wing.mean_aerodynamic_chord() + 1e-8)
        opti.subject_to([static_margin >= 0.05, static_margin <= 0.19])

        # Stability derivative boxes
        Cma_expr = aero.get("Cma")
        Cnb_expr = aero.get("Cn_beta", aero.get("Cnb", None))
        Clp_expr = aero.get("Cl_p",    aero.get("Clp", None))
        Cmq_expr = aero.get("Cmq")
        Cnr_expr = aero.get("Cn_r",    aero.get("Cnr", None))
        Clb_expr = aero.get("Cl_beta", aero.get("Clb", None))
        stab_box_constraints = []
        if Cma_expr is not None: stab_box_constraints += [Cma_expr <= -0.1, Cma_expr >= -2.0]
        if Cnb_expr is not None: stab_box_constraints += [Cnb_expr >= 0.03, Cnb_expr <= 0.1]
        if Clp_expr is not None: stab_box_constraints += [Clp_expr >= -0.6, Clp_expr <= -0.30]
        if Cmq_expr is not None: stab_box_constraints += [Cmq_expr >= -6.00, Cmq_expr <= -2.00]
        if Cnr_expr is not None: stab_box_constraints += [Cnr_expr >= -0.1, Cnr_expr <= -0.02]
        if Clb_expr is not None: stab_box_constraints += [Clb_expr <= -0.02, Clb_expr >= -0.20]
        opti.subject_to(stab_box_constraints)

        # ---- Cm must be negative and decreasing over ±STAB_BAND_DEG around cruise ----
        STAB_BAND_DEG = 3
        CMA_SIGN_MARGIN = 0.003
        band_deg = STAB_BAND_DEG  # e.g. 7 => enforce from αc-7° .. αc+7°
        N_pts = 5  # odd count works nicely; dense enough to be robust

        alpha_lo = alpha_cruise - band_deg
        alpha_hi = alpha_cruise + band_deg
        dalpha_deg = (alpha_hi - alpha_lo) / (N_pts - 1)  # symbolic (CasADi) step in degrees
        dalpha_rad = dalpha_deg * (np.pi / 180)  # symbolic step in radians

        # Convert slope threshold from "per degree" to "per radian" (CORRECT: divide by π/180)
        CMA_BAND_THRESH_PER_DEG = -0.003  # your setting: ΔCm per degree (must be negative)
        CMA_BAND_THRESH_PER_RAD = CMA_BAND_THRESH_PER_DEG / (np.pi / 180)

        # Build Cm(α) samples symbolically
        Cm_vals = []
        for i in range(N_pts):  # iterate Python ints, safe
            a_i = alpha_lo + i * dalpha_deg  # symbolic alpha in deg
            Cm_i = asb.AeroBuildup(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=op_point.velocity, alpha=a_i),
                xyz_ref=mass_props_TOGW.xyz_cg
            ).run()["Cm"]
            Cm_vals.append(Cm_i)
        Cm_vals = cas.vcat(Cm_vals)  # CasADi column vector

        # (1) Cm positive at the low end and negative at the high end (sign guard)
        opti.subject_to(Cm_vals[0] >= +CMA_SIGN_MARGIN)
        opti.subject_to(Cm_vals[-1] <= -CMA_SIGN_MARGIN)

        # (2) Monotone decreasing with a max (negative) slope at every segment
        for i in range(N_pts - 1):
            dCm = Cm_vals[i + 1] - Cm_vals[i]
            opti.subject_to(dCm / dalpha_rad <= CMA_BAND_THRESH_PER_RAD)

        # --- Min-sink evaluation at Vcruise
        ab_ms = asb.AeroBuildup(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=Vcruise, alpha=alpha_ms),
            xyz_ref=mass_props_TOGW.xyz_cg
        ).run()
        CL_ms = ab_ms["CL"]; CD_ms = ab_ms["CD"]
        opti.subject_to(CL_ms >= 0.2)
        Vsink = np.sqrt(2 * W / (RHO * (S + 1e-12))) * (CD_ms / (CL_ms ** 1.5 + 1e-9))

        # Induced drag estimate
        CDi = (aero["CL"] ** 2) / (np.pi * E_OSWALD * (AR + 1e-9))

        # ----- Geometry placement (match teammate)
        opti.subject_to([
            cylinder_center_x - CYL_LEN / 2 >= boom_nose_x,   # pod fully on boom (after boom nose)
            cylinder_center_x + CYL_LEN / 2 <= x_tail,        # pod ends before tail start
            (cylinder_center_x + CYL_LEN / 2) <= 0.0,         # pod ends in front of x=0
        ])

        # Tail volume-ish guards
        tail_moment_arm = vtail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]
        opti.subject_to(vtail.area() * (np.sind(vtail_dihedral) ** 2) * tail_moment_arm /
                        (wing.area() * wing.mean_aerodynamic_chord()) > 0.25)
        opti.subject_to(vtail.area() * (np.cosd(vtail_dihedral) ** 2) * tail_moment_arm /
                        (wing.area() * wing.span()) > 0.01)

        # ----- Objective (no mass penalty)
        J = (LD_WEIGHT * LD
             - SINK_WEIGHT * Vsink
             - STALL_WEIGHT * Vstall
             - INDUCED_PENALTY_WEIGHT * CDi
             - MASS_WEIGHT * mass_props_TOGW.mass)
        opti.maximize(J)

        # IPOPT verbosity
        opti.solver("ipopt", {
            "ipopt.print_level": 5,
            "print_time": True,
        })

        # ----- Solve
        config_name = f"{CONFIG_NAME_BASE}__wing-{wing_airfoil_name}__vtail-{vtail_airfoil_name}"
        sol = opti.solve(verbose=True)

        # ===================== OBJECTIVE PUSH DIAGNOSTICS =====================
        # Build numeric vals dict from the solved design
        vals = {
            # identify airfoils used for this run
            "wing_airfoil_name": wing_airfoil_name,
            "vtail_airfoil_name": vtail_airfoil_name,

            # geometry
            "wing_span": float(sol(wing_span)),
            "wing_root_chord": float(sol(wing_root_chord)),
            "wing_taper_ratio": float(sol(wing_taper_ratio)),
            "winglet_height_fraction": float(sol(winglet_height_fraction)),
            "winglet_cant_angle": float(sol(winglet_cant_angle)),
            "winglet_sweep_angle": float(sol(winglet_sweep_angle)),
            "winglet_taper_ratio": float(sol(winglet_taper_ratio)),
            "vtail_span": float(sol(vtail_span)),
            "vtail_root_chord": float(sol(vtail_root_chord)),
            "vtail_taper_ratio": float(sol(vtail_taper_ratio)),
            "vtail_dihedral": float(sol(vtail_dihedral)),
            "vtail_le_sweep": float(sol(vtail_le_sweep)),
            "i_tail_deg": float(sol(i_tail_deg)),
            "boom_nose_x": float(sol(boom_nose_x)),
            "x_tail": float(sol(x_tail)),
            "cylinder_center_x": float(sol(cylinder_center_x)),
            "engine_span_fraction": float(sol(engine_span_fraction)),
            "wing_motor_x_frac": float(sol(wing_motor_x_frac)),

            # flight / mass
            "alpha_ms": float(sol(alpha_ms)),
            "mass_total": float(sol(mass_props_TOGW.mass)),
            "xcg_total": float(sol(mass_props_TOGW.x_cg)),
        }

        # Small helper to compute all 4 terms and J using your current weights
        def _eval_all_terms(vals):
            out = _evaluate_terms_numeric(
                vals,
                airfoils=airfoils, E_OSWALD=E_OSWALD, RHO=RHO,
                CRUISE_TO_VSTALL=CRUISE_TO_VSTALL, CLMAX_MARGIN=CLMAX_MARGIN,
                WING_CLMAX_LOOKUP=WING_CLMAX_LOOKUP
            )
            LD = out["LD"]
            Vsink = out["Vsink"]
            Vstall = out["Vstall"]
            CDi = out["CDi"]
            J = (LD_WEIGHT * LD
                 - SINK_WEIGHT * Vsink
                 - STALL_WEIGHT * Vstall
                 - INDUCED_PENALTY_WEIGHT * CDi)
            return dict(LD=LD, Vsink=Vsink, Vstall=Vstall, CDi=CDi, J=J)

        # Variables we allow to move for the *direction* computation (with scales that mirror your Opti scales)
        probe = [
            ("wing_span", 1.0),
            ("wing_root_chord", 0.1),
            ("wing_taper_ratio", 1.0),
            ("x_tail", 1.0),
            ("boom_nose_x", 0.1),
            ("cylinder_center_x", 0.1),
            ("vtail_span", 0.1),
            ("vtail_root_chord", 0.1),
            ("vtail_taper_ratio", 1.0),
            ("vtail_dihedral", 10.0),
            ("engine_span_fraction", 1.0),
            ("winglet_height_fraction", 0.1),
            ("winglet_cant_angle", 100.0),
            ("winglet_sweep_angle", 10.0),
            ("winglet_taper_ratio", 1.0),
        ]

        # 1) Numeric gradient of J wrt these vars (scaled)
        EPS = 1e-3  # FD step as a fraction of "scale"
        base_terms = _eval_all_terms(vals)
        gJ = []
        for name, scale in probe:
            h = EPS * scale
            vals_p = vals.copy();
            vals_p[name] = vals[name] + h
            Jp = _eval_all_terms(vals_p)["J"]
            dJ_dvar_scaled = (Jp - base_terms["J"]) / (h + 1e-20) * scale
            gJ.append(dJ_dvar_scaled)

        # Normalize ascent direction of J in (scaled) variable space
        import numpy as _np
        gJ = _np.array(gJ, dtype=float)
        norm_gJ = float(_np.linalg.norm(gJ, 2))

        if norm_gJ < 1e-12:
            print("\n### Objective push: gradient near zero (design at a flat spot).")
        else:
            d_hat = gJ / norm_gJ  # unit ascent direction for J

            # 2) Directional derivative of each term along +∇J
            H = 5e-3  # small move along the unit direction
            vals_step = vals.copy()
            for (name, scale), comp in zip(probe, d_hat):
                vals_step[name] = vals[name] + (H * comp) / (scale + 1e-20)

            terms_step = _eval_all_terms(vals_step)
            push = {
                "LD": (terms_step["LD"] - base_terms["LD"]) / (H + 1e-20),
                "Vsink": (terms_step["Vsink"] - base_terms["Vsink"]) / (H + 1e-20),
                "Vstall": (terms_step["Vstall"] - base_terms["Vstall"]) / (H + 1e-20),
                "CDi": (terms_step["CDi"] - base_terms["CDi"]) / (H + 1e-20),
            }

            print("\n### How the objective is pushing each term (directional derivative along +∇J):")
            print("  (Units: term-units per unit move along the steepest-ascent direction of J)")
            print(
                f"  LD push:     {push['LD']:+.4g}   (objective ascent tends to {'increase' if push['LD'] > 0 else 'decrease' if push['LD'] < 0 else 'not change'} LD)")
            print(
                f"  Vsink push:  {push['Vsink']:+.4g} (want negative; objective ascent tends to {'increase' if push['Vsink'] > 0 else 'decrease' if push['Vsink'] < 0 else 'not change'} Vsink)")
            print(
                f"  Vstall push: {push['Vstall']:+.4g} (want negative; objective ascent tends to {'increase' if push['Vstall'] > 0 else 'decrease' if push['Vstall'] < 0 else 'not change'} Vstall)")
            print(
                f"  CDi push:    {push['CDi']:+.4g}   (want negative; objective ascent tends to {'increase' if push['CDi'] > 0 else 'decrease' if push['CDi'] < 0 else 'not change'} CDi)")

        # 3) Also print ∂J/∂weights (weight leverage) — four clean numbers
        print("\n### ∂J/∂(weights) at solution (leverage if you tweak weights):")
        print(f"  ∂J/∂LD_WEIGHT                      = {base_terms['LD']:.5g}")
        print(f"  ∂J/∂SINK_WEIGHT   (note: negative) = {-base_terms['Vsink']:.5g}")
        print(f"  ∂J/∂STALL_WEIGHT  (note: negative) = {-base_terms['Vstall']:.5g}")
        print(f"  ∂J/∂INDUCED_PENALTY_WEIGHT (neg)   = {-base_terms['CDi']:.5g}")
        # =================== END OBJECTIVE PUSH DIAGNOSTICS =====================

        # Values to print
        boom_len_val = float(sol(x_tail) - sol(boom_nose_x))
        print(f"Boom length: {boom_len_val:.4f} m")

        # If you added i_tail_deg (incidence) as a variable:
        try:
            print(f"Tail incidence (deg): {float(sol(i_tail_deg)):.3f}")
        except Exception:
            pass  # not present

        print(f"Tail AR (-): {float(sol(AR_tail)):.3f}")

        # Reporting
        report_solution(
            sol=sol, config_name=config_name,
            airplane=sol(airplane), op_point=sol(op_point),
            mass_props=sol(mass_props), mass_props_TOGW=sol(mass_props_TOGW),
            LD_expr=sol(LD), aero=sol(aero), CDi=sol(CDi),
            opti_vars={k: sol(v) for k, v in {
                "Cruise Speed (m/s)"        : Vcruise,
                "Stall Speed (m/s)"         : Vstall,
                "Cruise AoA (deg)"          : alpha_cruise,
                "Min Sink AoA (deg)"        : alpha_ms,
                "Wing AR (-)"               : AR,
                "Boom Nose Position (m)"    : boom_nose_x,
                "Cylinder Center (m)"       : cylinder_center_x,
                "Tail Position (m)"         : x_tail,
                "Boom Radius (m)"           : boom_radius,
                "Engine Span Fraction"      : engine_span_fraction,
                "Payload Mass (kg)"         : payload_mass,
                "Wing Span (m)"             : wing_span,
                "Wing Root Chord (m)"       : wing_root_chord,
                "Wing Taper Ratio"          : wing_taper_ratio,
                "Winglet Height Fraction"   : winglet_height_fraction,
                "Winglet Cant Angle (deg)"  : winglet_cant_angle,
                "Winglet Sweep Angle (deg)" : winglet_sweep_angle,
                "Winglet Taper Ratio"       : winglet_taper_ratio,
                "V-Tail Span (m)"           : vtail_span,
                "V-Tail Root Chord (m)"     : vtail_root_chord,
                "V-Tail Taper Ratio"        : vtail_taper_ratio,
                "V-Tail Dihedral (deg)"     : vtail_dihedral,
                "V-Tail LE Sweep (deg)"     : vtail_le_sweep,
                "x_cg Base Electronics (m)" : x_cg_base_electronics,
                "x_cg VTOL Electronics (m)" : x_cg_vtol_electronics,
                "x_cg Non-VTOL Electronics (m)": x_cg_non_vtol_electronics,
                "x_cg CV Electronics (m)"   : x_cg_cv_electronics,
                "x_cg Rear Engine (m)"      : x_cg_rear_engine,
                "x_cg Payload (m)"          : x_cg_payload,
            }.items()},
            Vsink=sol(Vsink), Vstall=sol(Vstall),
            boom_nose_x=sol(boom_nose_x), x_tail=sol(x_tail), wing=sol(wing)
        )

        result = dict(
            config_name=config_name,
            wing_af=wing_airfoil_name,
            vtail_af=vtail_airfoil_name,
            J=float(sol(J)),
            LD=float(sol(LD)),
            mass=float(sol(mass_props_TOGW.mass)),
            Vsink=float(sol(Vsink)),
            Vstall=float(sol(Vstall)),
        )
        return True, result

    except Exception as e:
        print(f"[{wing_airfoil_name} | {vtail_airfoil_name}] solve failed: {e}")
        traceback.print_exc()
        return False, {"wing_af": wing_airfoil_name, "vtail_af": vtail_airfoil_name, "error": str(e)}

# ----------------- Sweep driver -----------------
def sweep_airfoils():
    best = None
    results = []
    for wing_af, vtail_af in itertools.product(AIRFOILS_WING, AIRFOILS_VTAIL):
        print("\n" + "="*80)
        print(f"Solving for airfoil combo: Wing={wing_af}, V-Tail={vtail_af}")
        ok, res = build_and_solve(wing_af, vtail_af)
        results.append((ok, res))
        if ok:
            if (best is None) or (res["J"] > best["J"]):
                best = res

    print("\n" + "#"*90)
    print("# Sweep complete. Summary (sorted by total objective J):")
    ranked = [r for ok, r in results if ok]
    ranked.sort(key=lambda r: r["J"], reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"{i:2d}. J={r['J']:.3f} | LD={r['LD']:.2f} | Mass={r['mass']:.3f} kg | "
              f"Vsink={r['Vsink']:.2f} m/s | Vstall={r['Vstall']:.2f} m/s | "
              f"Wing={r['wing_af']} | VTail={r['vtail_af']} | {r['config_name']}")

    if best is None:
        print("No successful solves. Consider loosening bounds, changing initial guesses, or re-tuning weights.")
    else:
        print("\nBest airfoil combo:")
        print(f"  -> Wing = {best['wing_af']}, V-Tail = {best['vtail_af']}")
        print(f"  -> J = {best['J']:.3f}, L/D = {best['LD']:.2f}, Mass = {best['mass']:.3f} kg, "
              f"Vsink = {best['Vsink']:.2f} m/s, Vstall = {best['Vstall']:.2f} m/s")
    print("#"*90 + "\n")

if __name__ == "__main__":
    sweep_airfoils()
