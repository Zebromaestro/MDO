import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy
from pathlib import Path


opti = asb.Opti(
    # variable_categories_to_freeze="all",
    # freeze_style="float"def
)

make_plots = True

##### Section: Parameters

# wing_method = '3d printing'
wing_method = 'foam'

wing_span = 1
wing_dihedral_angle_deg = 20

airfoils = {
    name: asb.Airfoil(
        name=name,
    ) for name in [
        "ag04",
        # "ag09",
        "ag13",
        "naca0008"
    ]
}

for v in airfoils.values():
    v.generate_polars(
        cache_filename=f"cache/{v.name}.json",
        alphas=np.linspace(-10, 20, 21)
    )

##### Section: Vehicle Overall Specs

op_point = asb.OperatingPoint(
    velocity=opti.variable(
        init_guess=14,
        lower_bound=1,
        log_transform=True
    ),
    alpha=opti.variable(
        init_guess=0,
        lower_bound=-10,
        upper_bound=20
    )
)

design_mass_TOGW = opti.variable(
    init_guess=0.1,
    lower_bound=1e-3
)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3)

LD_cruise = opti.variable(
    init_guess=15,
    lower_bound=0.1,
    log_transform=True
)

g = 9.81

target_climb_angle = 30  # degrees

thrust_cruise = (
        design_mass_TOGW * g / LD_cruise
)

thrust_climb = (
        design_mass_TOGW * g / LD_cruise +
        design_mass_TOGW * g * np.sind(target_climb_angle)
)

##### Section: Vehicle Definition

"""
Coordinate system:

Geometry axes. Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main
wing.

"""

# ##### NEW: Define x-stations for the nose and tail #####
x_nose = opti.variable(
    init_guess=-0.1,
    upper_bound=0,
)
x_tail = opti.variable(
    init_guess=0.5,
    lower_bound=0.1,
)

### Wing
wing_root_chord = opti.variable(
    init_guess=0.15,
    lower_bound=1e-3
)

# ##### MODIFIED: Define a wing with a flat inner section and dihedral/tapered outer section #####
wing_center_span_fraction = 0.4 # The fraction of the wing that is the flat, untapered center section
wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec( # Root
            xyz_le=[-0.25 * wing_root_chord, 0, 0],
            chord=wing_root_chord,
            airfoil=airfoils["ag13"],
        ),
        asb.WingXSec( # Dihedral break
            xyz_le=[-0.25 * wing_root_chord, wing_span / 2 * wing_center_span_fraction, 0],
            chord=wing_root_chord,
            airfoil=airfoils["ag13"],
        ),
        asb.WingXSec( # Tip
            xyz_le=[
                -0.25 * wing_root_chord * 0.7,
                wing_span / 2,
                wing_span / 2 * (1 - wing_center_span_fraction) * np.tand(wing_dihedral_angle_deg)
            ],
            chord=wing_root_chord * 0.7,
            airfoil=airfoils["ag13"],
        )
    ]
)


# ##### H-Tail Definition #####
# --- Define Horizontal Stabilizer ---
h_tail_span = 0.35 * wing_span
h_tail_root_chord = 0.08
h_tail = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=h_tail_root_chord,
            airfoil=airfoils["naca0008"]
        ),
        asb.WingXSec(
            xyz_le=[0.02, h_tail_span / 2, 0],
            chord=h_tail_root_chord * 0.8,
            airfoil=airfoils["naca0008"]
        ),
    ]
).translate([
    x_tail,  # ##### EDIT: Position tail at the x_tail location #####
    0,
    0
])

# --- Define a single Vertical Stabilizer ---
v_tail_height = 0.12
v_tail_root_chord = 0.09
v_tail = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=v_tail_root_chord,
            airfoil=airfoils["naca0008"]
        ),
        asb.WingXSec(
            xyz_le=[0.03, 0, v_tail_height],
            chord=v_tail_root_chord * 0.7,
            airfoil=airfoils["naca0008"]
        )
    ]
)

# --- Place the two Vertical Stabilizers at the tips of the H-tail ---
v_tail_right = v_tail.translate([
    x_tail + 0.01, # ##### EDIT: Position relative to x_tail #####
    h_tail_span / 2,
    0
])
v_tail_right.name = "Right Vertical Stabilizer"

v_tail_left = v_tail.translate([
    x_tail + 0.01, # ##### EDIT: Position relative to x_tail #####
    -h_tail_span / 2,
    0
])
v_tail_left.name = "Left Vertical Stabilizer"

# ##### MODIFIED: Define the Fuselage as a pod-and-boom #####
x_pod_end = x_nose + 8 * u.inch

fuselage = asb.Fuselage(
    name="Fuse",
    xsecs=[
        asb.FuselageXSec( # Start of rectangular pod
            xyz_c=[x_nose, 0, 0],
            height=1.5 * u.inch,
            width=2.0 * u.inch,
        ),
        asb.FuselageXSec( # End of rectangular pod
            xyz_c=[x_pod_end, 0, 0],
            height=1.5 * u.inch,
            width=2.0 * u.inch,
        ),
        asb.FuselageXSec( # Start of boom
            xyz_c=[x_pod_end, 0, 0],
            radius=7e-3 / 2
        ),
        asb.FuselageXSec( # End of boom
            xyz_c=[x_tail, 0, 0],
            radius=7e-3 / 2
        )
    ]
)

# ##### EDIT: Update airplane definition to include the fuselage #####
airplane = asb.Airplane(
    name="Feather Flying Wing H-Tail",
    wings=[
        wing,
        h_tail,
        v_tail_right,
        v_tail_left
    ],
    fuselages=[fuselage]
)

##### Section: Internal Geometry and Weights

mass_props = {}

### Lifting bodies
if wing_method == '3d printing':
    raise ValueError

elif wing_method == 'foam':
    density = 2 * u.lbm / u.foot ** 3
    mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
        mass=wing.volume() * density,
        x_cg=0,  # At the wing datum
        z_cg=(0.03591) * (
                np.sind(wing_dihedral_angle_deg) / np.sind(11)
        ) * (
                     wing_span / 1
             ),
    )
elif wing_method == 'elf':
    raise ValueError

total_tail_volume = h_tail.volume() + v_tail_right.volume() + v_tail_left.volume()
mass_props["tail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass=total_tail_volume * 80,
    x_cg=x_tail + 0.50 * h_tail_root_chord,
)

mass_props["linkages"] = asb.MassProperties(
    mass=1e-3,
    x_cg=(x_nose + x_tail) / 2 # ##### EDIT: Positioned along boom
)

# ##### EDIT: Reposition avionics relative to the new nose location #####
mass_props["motor"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.49e-3,
    x_cg=x_nose - 0.3 * u.inch
)
mass_props["motor_bolts"] = asb.mass_properties_from_radius_of_gyration(
    mass=4 * 0.075e-3,
    x_cg=x_nose
)

mass_props["propeller"] = asb.mass_properties_from_radius_of_gyration(
    mass=1.54e-3,
    x_cg=x_nose - 0.7 * u.inch
)

mass_props["propeller_band"] = asb.mass_properties_from_radius_of_gyration(
    mass=0.06e-3,
    x_cg=mass_props["propeller"].x_cg
)

mass_props["flight_computer"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.30e-3,
    x_cg=x_nose + 2 * u.inch + (1.3 * u.inch) / 2
)

mass_props["battery"] = asb.mass_properties_from_radius_of_gyration(
    mass=4.61e-3,
    x_cg=x_nose + 2 * u.inch
)

# ##### NEW: Add mass for the pod structure #####
mass_props["pod_structure"] = asb.MassProperties(
    mass=10e-3, # Assumed 10g pod
    x_cg=(x_nose + x_pod_end) / 2
)


# ##### MODIFIED: Adjust boom mass to account for pod #####
boom_length = np.maximum(0, x_tail - x_pod_end)
mass_props["boom"] = asb.mass_properties_from_radius_of_gyration(
    mass=7.0e-3 * (boom_length / 826e-3),
    x_cg=(x_pod_end + x_tail) / 2
)

mass_props["ballast"] = asb.MassProperties(
    mass=opti.variable(init_guess=0, lower_bound=0),
    x_cg=opti.variable(init_guess=0, lower_bound=x_nose, upper_bound=x_tail),
)

### Summation
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

### Add glue weight
mass_props['glue_weight'] = mass_props_TOGW * 0.08
mass_props_TOGW += mass_props['glue_weight']

##### Section: Aerodynamics

ab = asb.AeroBuildup(
    airplane=airplane,
    op_point=op_point,
    xyz_ref=mass_props_TOGW.xyz_cg
)
aero = ab.run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False,
)

opti.subject_to([
    aero["L"] >= 9.81 * mass_props_TOGW.mass,
    aero["Cm"] == 0,
])

LD = aero["L"] / aero["D"]
power_loss = aero["D"] * op_point.velocity
sink_rate = power_loss / 9.81 / mass_props_TOGW.mass

##### Section: Stability
static_margin = (aero["x_np"] - mass_props_TOGW.x_cg) / wing.mean_aerodynamic_chord()

opti.subject_to(
    static_margin == 0.08
)

##### Section: Finalize Optimization Problem
objective = sink_rate / 0.3 + mass_props_TOGW.mass / 0.100 * 0.1
penalty = (mass_props["ballast"].x_cg / 1e3) ** 2

opti.minimize(objective + penalty)

### Additional constraint
opti.subject_to([
    LD_cruise == LD,
    design_mass_TOGW == mass_props_TOGW.mass
])

# ##### EDIT: Update tail volume and add fuselage constraints #####
tail_moment_arm = h_tail.aerodynamic_center(chord_fraction=0.25)[0] - mass_props_TOGW.xyz_cg[0]

# Horizontal tail volume coefficient
opti.subject_to(
    h_tail.area() * tail_moment_arm / (wing.area() * wing.mean_aerodynamic_chord()) > 0.4
)
# Vertical tail volume coefficient
total_vtail_area = v_tail_right.area() + v_tail_left.area()
opti.subject_to(
    total_vtail_area * tail_moment_arm / (wing.area() * wing.span()) > 0.02
)

# ##### NEW: Fuselage and propeller constraints #####
opti.subject_to([
    x_nose < -0.25 * wing_root_chord - 0.5 * u.inch,  # propeller must extend in front of wing
    x_tail - x_nose < 0.826,  # boom length constraint
])


if __name__ == '__main__':
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
    s = lambda x: sol.value(x)

    # Substitute numerical values into all of the objects
    airplane = sol(airplane)
    op_point = sol(op_point)
    mass_props = sol(mass_props)
    mass_props_TOGW = sol(mass_props_TOGW)
    aero = sol(aero)

    # --- START OF EDITED SECTION ---
    # Create a copy of the SOLVED airplane to modify for the AVL run.
    avl_airplane = copy.deepcopy(airplane)

    # Get the main wing from this new, solved airplane object.
    wing_lowres = avl_airplane.wings[0]

    # Your original logic for reducing the number of cross-sections is fine.
    xsecs_to_keep = np.arange(len(wing_lowres.xsecs)) % 2 == 0
    xsecs_to_keep[0] = True
    xsecs_to_keep[-1] = True
    wing_lowres.xsecs = np.array(wing_lowres.xsecs)[xsecs_to_keep]

    try:
        avl_aero = asb.AVL(
            # Pass the MODIFIED, SOLVED airplane object here.
            airplane=avl_airplane,
            op_point=op_point,
            xyz_ref=mass_props_TOGW.xyz_cg
        ).run()
    except (FileNotFoundError, AttributeError):  # Added AttributeError for safety
        class EmptyDict:
            def __getitem__(self, item):
                return "Install AVL to see this."

        avl_aero = EmptyDict()
    # --- END OF EDITED SECTION ---

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    ##### Section: Printout
    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))


    def fmt(x):
        try:
            return f"{s(x):.6g}"
        except:
            return "N/A"


    print_title("Outputs")
    for k, v in {
        "mass_TOGW"            : f"{fmt(mass_props_TOGW.mass)} kg ({fmt(mass_props_TOGW.mass / u.lbm)} lbm)",
        "L/D (actual)"         : fmt(LD_cruise),
        "Cruise Airspeed"      : f"{fmt(op_point.velocity)} m/s",
        "Cruise AoA"           : f"{fmt(op_point.alpha)} deg",
        "Cruise CL"            : fmt(aero['CL']),
        "Sink Rate"            : fmt(sink_rate),
        "Cma"                  : fmt(aero['Cma']),
        "Cnb"                  : fmt(aero['Cnb']),
        "Cm"                   : fmt(aero['Cm']),
        "Wing Reynolds Number" : eng_string(op_point.reynolds(sol(wing.mean_aerodynamic_chord()))),
        "AVL: Cma"             : avl_aero['Cma'],
        "AVL: Cnb"             : avl_aero['Cnb'],
        "AVL: Cm"              : avl_aero['Cm'],
        "AVL: Clb Cnr / Clr Cnb": avl_aero['Clb Cnr / Clr Cnb'],
        "CG location"          : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
        "Wing Span"            : f"{fmt(wing_span)} m ({fmt(wing_span / u.foot)} ft)",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    fmtpow = lambda x: fmt(x) + " W"

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {s(v.mass) * 1e3:.2f} g ({s(v.mass) / u.oz:.2f} oz)")

    if make_plots:
        ##### Section: Geometry
        airplane.draw_three_view(show=False)
        p.show_plot(tight_layout=False, savefig="figures/three_view.png")

        ##### Section: Mass Budget
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(aspect="equal"), dpi=300)

        name_remaps = {
            **{
                k: k.replace("_", " ").title()
                for k in mass_props.keys()
            },
        }

        mass_props_to_plot = sol(copy.deepcopy(mass_props))
        if "ballast" in mass_props_to_plot and mass_props_to_plot["ballast"].mass < 1e-6:
            mass_props_to_plot.pop("ballast")
        p.pie(
            values=[
                v.mass
                for v in mass_props_to_plot.values()
            ],
            names=[
                n if n not in name_remaps.keys() else name_remaps[n]
                for n in mass_props_to_plot.keys()
            ],
            center_text=f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass * 1e3):.2f} g",
            label_format=lambda name, value, percentage: f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%",
            startangle=110,
            arm_length=30,
            arm_radius=20,
            y_max_labels=1.1
        )
        p.show_plot(savefig="figures/mass_budget.png")
