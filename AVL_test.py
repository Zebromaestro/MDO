import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
from shutil import which

# --- Prerequisite Check ---
# Checks if the AVL executable is found in your system's PATH.
avl_is_present = which('avl') is not None
if not avl_is_present:
    raise FileNotFoundError(
        "AVL executable not found in PATH. Please install AVL and ensure it's accessible."
    )

# --- 1. Define Airplane Geometry (with Control Surfaces) ---
sd7037 = asb.Airfoil("sd7037")
naca0012 = asb.Airfoil("naca0012")  # A common airfoil for the tail

airplane = asb.Airplane(
    name="Vanilla",
    xyz_ref=[0.5, 0, 0],
    s_ref=9,
    c_ref=0.9,
    b_ref=10,
    wings=[
        # Main Wing
        asb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=2,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0.2, 5, 1],
                    chord=0.6,
                    twist=2,
                    airfoil=sd7037,
                    control_surface_type='symmetric',
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                )
            ]
        ),
        # Horizontal Stabilizer
        asb.Wing(
            name="H-stab",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=naca0012,
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 1.25, 0],
                    chord=0.42,
                    airfoil=naca0012,
                    control_surface_type='symmetric',
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
            ]
        ).translate([4, 0, 0]),
        # Vertical Stabilizer
        asb.Wing(
            name="V-stab",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=naca0012,
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 0, 1],
                    chord=0.42,
                    airfoil=naca0012,
                    control_surface_type='symmetric',
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                )
            ]
        ).translate([4, 0, 0])
    ]
)

# --- 2. Analysis Setup ---
# Define a baseline operating point for sweeps
op_point_base = asb.OperatingPoint(
    velocity=30,  # m/s
    alpha=3,      # A typical cruise angle of attack in degrees
)

# Define the ranges for our sweeps
alpha_sweep = np.linspace(-5, 15, 21)
beta_sweep = np.linspace(-15, 15, 21)
deflection_sweep = np.linspace(-20, 20, 21)

# --- 3. Run Analysis Sweeps (with Debug Prints) ---

# A) Performance and Longitudinal Stability (Alpha Sweep)
print("Running alpha sweep for performance and longitudinal stability...")
results_alpha_sweep = {"CL": [], "CD": [], "Cm": [], "L/D": []}
for alpha in alpha_sweep:
    print(f"  Analyzing alpha = {alpha:.2f} deg...")
    op_point = op_point_base.copy()
    op_point.alpha = alpha
    aero = asb.AVL(airplane=airplane, op_point=op_point).run()
    results_alpha_sweep["CL"].append(aero["CL"])
    results_alpha_sweep["CD"].append(aero["CD"])
    results_alpha_sweep["Cm"].append(aero["Cm"])
    results_alpha_sweep["L/D"].append(aero["CL"] / aero["CD"] if aero["CD"] > 0 else 0)

# B) Lateral-Directional Static Stability (Beta Sweep)
print("\nRunning beta sweep for lateral-directional stability...")
results_beta_sweep = {"Cn": [], "Cl": []}
for beta in beta_sweep:
    print(f"  Analyzing beta = {beta:.2f} deg...")
    op_point = op_point_base.copy()
    op_point.beta = beta
    aero = asb.AVL(airplane=airplane, op_point=op_point).run()
    results_beta_sweep["Cn"].append(aero["Cn"])
    results_beta_sweep["Cl"].append(aero["Cl"])

# C) Control Power (Deflection Sweeps) - ALL REMOVED
print("\nSkipping deflection sweeps for control power due to errors...")

# D) Dynamic Damping Derivatives
print("\nCalculating dynamic damping derivatives...")
dynamic_derivs = asb.AVL(airplane=airplane, op_point=op_point_base, verbose=False).run()
print("\n--- Dynamic Stability Derivatives ---")
print(f"  Roll Damping (Clp): {dynamic_derivs['Clp']:.4f}")
print(f"  Pitch Damping (Cmq): {dynamic_derivs['Cmq']:.4f}")
print(f"  Yaw Damping (Cnr): {dynamic_derivs['Cnr']:.4f}")
print("-------------------------------------\n")


# --- 4. Plotting Results ---
print("Generating plots...")
fig, axs = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('AVL Aerodynamic Analysis', fontsize=20)

# Plot 1: Lift Curve
ax = axs[0, 0]
ax.plot(alpha_sweep, results_alpha_sweep["CL"])
ax.set_xlabel("Angle of Attack α (°)")
ax.set_ylabel("Lift Coefficient (CL)")
ax.set_title("Lift Curve")
ax.grid(True)

# Plot 2: Drag Polar
ax = axs[0, 1]
ax.plot(results_alpha_sweep["CL"], results_alpha_sweep["CD"])
ax.set_xlabel("Lift Coefficient (CL)")
ax.set_ylabel("Drag Coefficient (CD)")
ax.set_title("Drag Polar")
ax.grid(True)

# Plot 3: Aerodynamic Efficiency
ax = axs[0, 2]
ax.plot(alpha_sweep, results_alpha_sweep["L/D"])
ax.set_xlabel("Angle of Attack α (°)")
ax.set_ylabel("Lift-to-Drag Ratio (L/D)")
ax.set_title("Aerodynamic Efficiency")
ax.grid(True)

# Plot 4: Longitudinal Stability
ax = axs[0, 3]
ax.plot(alpha_sweep, results_alpha_sweep["Cm"])
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel("Angle of Attack α (°)")
ax.set_ylabel("Pitching Moment (Cm)")
ax.set_title("Longitudinal Stability (Cm vs. α)")
ax.grid(True)

# Plot 5: Directional Stability
ax = axs[1, 0]
ax.plot(beta_sweep, results_beta_sweep["Cn"])
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel("Sideslip Angle β (°)")
ax.set_ylabel("Yawing Moment (Cn)")
ax.set_title("Directional Stability (Cn vs. β)")
ax.grid(True)

# Plot 6: Lateral Stability
ax = axs[1, 1]
ax.plot(beta_sweep, results_beta_sweep["Cl"])
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel("Sideslip Angle β (°)")
ax.set_ylabel("Rolling Moment (Cl)")
ax.set_title("Lateral Stability (Cl vs. β)")
ax.grid(True)

# Plots 10, 11, 12: Dynamic Derivatives (as text)
ax = axs[2, 1]
ax.text(0.5, 0.6, "Dynamic Derivatives", ha='center', va='center', fontsize=14, weight='bold')
ax.text(0.5, 0.4,
    f"Roll Damping (Clp): {dynamic_derivs['Clp']:.4f}\n"
    f"Pitch Damping (Cmq): {dynamic_derivs['Cmq']:.4f}\n"
    f"Yaw Damping (Cnr): {dynamic_derivs['Cnr']:.4f}",
    ha='center', va='center', fontsize=12, family='monospace'
)
ax.axis('off')

# Hide unused subplots
axs[1, 2].axis('off') # Hiding the subplot for the removed aileron plot
axs[1, 3].axis('off') # Hiding the subplot for the removed elevator plot
axs[2, 0].axis('off') # Hiding the subplot for the removed rudder plot
axs[2, 2].axis('off')
axs[2, 3].axis('off')


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
