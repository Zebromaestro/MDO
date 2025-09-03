import aerosandbox as asb
import aerosandbox.numpy as np

# Here, all distances are in meters and all angles are in degrees.
airplane = asb.Airplane(
    name="Example Airplane",
    xyz_ref=[0.5, 0, 0],  # Reference for moments
    s_ref=9,  # Reference area
    c_ref=0.9,  # Reference chord
    b_ref=10,  # Reference span
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections, or "XSecs"
                asb.WingXSec(  # Root
                    xyz_le=[0, 0, 0],  # Coordinates of the XSec's leading edge
                    chord=1,
                    twist=1,  # in degrees
                    airfoil=asb.Airfoil("sd7032"),
                    control_surface_is_symmetric=False,  # Aileron
                    control_surface_deflection=0,  # in degrees
                    # (ctrl. surfs. are applied between this XSec and the next one.)
                ),
                asb.WingXSec(  # Tip
                    xyz_le=[0.2, 5, 1],
                    chord=0.6,
                    twist=-1,
                    airfoil=asb.Airfoil("sd7037"),
                )
            ]
        ),
        asb.Wing(
            name="H-stab",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=asb.Airfoil("ht08")
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 1.25, 0],
                    chord=0.42,
                    airfoil=asb.Airfoil("ht08")
                ),
            ]
        ).translate([4, 0, 0]), # Used to translate all cross sections of the wing.
        asb.Wing(
            name="V-stab",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=asb.Airfoil("ht08")
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 0, 1],
                    chord=0.42,
                    airfoil=asb.Airfoil("ht08")
                )
            ]
        ).translate([4, 0, 0]), # Used to translate all cross sections of the wing.
    ],
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[xi * 5 - 0.5, 0, 0],
                    radius=asb.Airfoil("naca0024").local_thickness(x_over_c=xi)
                )
                for xi in np.cosspace(0, 1, 30)
            ]
        )
    ]
)

airplane.draw_three_view()
