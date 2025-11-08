import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy

from aerosandbox.library.weights.raymer_cargo_transport_weights import mass_nacelles

opti = asb.Opti()
make_plots = True



wing_method = 'Monokote'

## CONSTANTS ##
density     = 1.225  # Note: AeroSandbox uses base SI units (or derived units thereof) everywhere, with no exceptions.
viscosity   = 1.81e-5  # So the units here are kg/(m*s)
weight      = ( 3.5 / 2.205 ) * 9.81  # N
wing_span   = 1
g           = 9.81
target_climb_angle = 45  # degrees


## What are we solving for? ##
aspect_ratio            = opti.variable(init_guess=10, lower_bound=0.1, upper_bound= 20, log_transform=True)
wing_area               = opti.variable(init_guess=1, log_transform=True)
Airfoils                = opti.variable(init_guess=1, log_transform=True)
Wing_Dihedral           = opti.variable(init_guess=1, log_transform=True)
LD_cruise               = opti.variable(init_guess=15, lower_bound=0.1, log_transform=True)
wing_dihedral_angle_deg = opti.variable(init_guess=15, lower_bound=0.1, upper_bound= 40, log_transform=True)


## Solving for Flight Conditions ##
op_point = asb.OperatingPoint(
    velocity=opti.variable(init_guess=10, lower_bound=1, log_transform=True),
    # This guesses initial flight v as 10 m/s

    alpha=opti.variable(init_guess=1, lower_bound=-10, upper_bound=10)
    # AoA is analyzed from -10 to 10
)

thrust_cruise = weight * g / LD_cruise
    # How much thrust you need at cruise (horizontal)

thrust_climb = weight * g / LD_cruise + weight * g * np.sind(target_climb_angle)
    # How much thrust to climb at climb angle (only adds the component of weight)



## GEOMETRY ##
    # (0,0,0) is at the quarter-chord of the main wing root. Geometry is all relative to that.
x_nose = opti.variable(init_guess=-0.1, upper_bound=1e-3)
x_tail = opti.variable(init_guess=0.7,  lower_bound=1e-3)
wing_root_chord = opti.variable(init_guess=0.15, lower_bound=1e-3)

def wing_rot(xyz):
    dihedral_rot = np.rotation_matrix_3D(angle=np.radians(wing_dihedral_angle_deg), axis="X")
    return dihedral_rot @ np.array(xyz)
    # Utility to tilt points by the wing dihedral angle around the X - axis

def wing_chord(y):
    spanfrac = y / (wing_span / 2)
    chordfrac = 1 - 0.4 * spanfrac - 0.47 * (1 - (1 - spanfrac ** 2 + 1e-16) ** 0.5)
        # c_over_c_root = 0.1 + 0.9 * (1 - (y / half_span) ** 2) ** 0.5
    return chordfrac * wing_root_chord

def wing_twist(y):
    return np.zeros_like(y)
    # Twist distribution

wing_ys = np.sinspace(0, wing_span / 2, 20, reverse_spacing=True)
    # Generates 11 y-locations from root to tip, clustered toward the root (sine spacing)


wing = asb.Wing(
    name="Main Wing", symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=wing_rot([-wing_chord(wing_ys[i]), wing_ys[i], 0]),
            chord=wing_chord(wing_ys[i]),
            airfoil=airfoils["ag13"],
            twist=wing_twist(wing_ys[i]),
        ) for i in range(np.length(wing_ys))
    ]
).translate([0.75 * wing_root_chord, 0, 0])
    # Generates the main wing
# FIGURE OUT HOW TO OPTIMIZE/ITERATE THIS TOO? 
