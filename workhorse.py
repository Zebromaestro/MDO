import aerosandbox as asb
import aerosandbox.numpy as np

opti = asb.Opti()

##### Constants

### Env. constants
g = 9.81       # m/s^2
mu = 1.775e-5  # kg/m/s (air viscosity)
rho = 1.23     # kg/m^3 (air density)

### Non-dimensional constants
C_Lmax = 0.7        # stall CL BY REARRANGING FOR CL
e = 0.92            # Oswald efficiency factor
k = 1.17            # form factor
N_ult = 3.3         # ultimate load factor      
S_wetratio = 2.075  # wetted area ratio (S_wet / S)
tau = 0.12          # airfoil thickness-to-chord ratio
W_W_coeff1 = 2e-5   # wing weight coefficient 1
W_W_coeff2 = 60     # wing weight coefficient 2

### Dimensional constants
V_min =10         # m/s (reference takeoff/stall speed check)
W_0 = 3.5*2.205          # N (aircraft weight excluding wing)

### Free variables (positive via log_transform)
AR = opti.variable(init_guess=10,    log_transform=True)  # aspect ratio
S  = opti.variable(init_guess=10,    log_transform=True)  # wing area, m^2
V  = opti.variable(init_guess=40,    log_transform=True)  # cruise speed, m/s
W  = opti.variable(init_guess=9000,  log_transform=True)  # total weight, N
C_L = opti.variable(init_guess=0.6,  log_transform=True)  # lift coefficient

### Practical bounds to avoid numerical pathologies / unphysical solutions
opti.subject_to([
    AR >= 1, AR <= 30,
    S  >= 1, S  <= 200,
    V  >= 1.3 * V_min,            # cruise above stall margin
    C_L >= 0.2, C_L <= 0.9 * C_Lmax,
    W  >= W_0                     # at least empty
])

### Wing weight model (no fuel terms)
W_w_surf = W_W_coeff2 * S
W_w_strc = W_W_coeff1 / tau * N_ult * AR ** 1.5 * np.sqrt(
    W_0 * W * S
)
W_w = W_w_surf + W_w_strc

### Entire weight
opti.subject_to(W >= W_0 + W_w)

### Aerodynamics / performance (no fuel; cruise lift balances W_0 + W_w)
W_cruise = W_0 + W_w
opti.subject_to(W_cruise <= 0.5 * rho * S * C_L * V ** 2)

### Stall / takeoff constraint at V_min with CLmax
opti.subject_to(W <= 0.5 * rho * S * C_Lmax * V_min ** 2)

### Drag build-up
eps = 1e-9
Re  = (rho / mu) * V * np.sqrt(S / AR) + eps
C_f = 0.074 / Re ** 0.2

# Fuselage drag: use a small fixed parasite coefficient (dimensionless) now that fuel volume is gone
C_D0_fuse = 0.01
C_D_fuse  = C_D0_fuse
C_D_wpar  = k * C_f * S_wetratio
C_D_ind   = C_L ** 2 / (np.pi * AR * e)
C_D       = C_D_fuse + C_D_wpar + C_D_ind
D         = 0.5 * rho * S * C_D * V ** 2

### Objective: without mission fuel, minimize total weight (equivalently, wing weight)
opti.minimize(W)

sol = opti.solve(max_iter=200)

# Report results
to_report = [
    ("V", V),
    ("W", W),
    ("C_L", C_L),
    ("AR", AR),
    ("S", S),
    ("C_D", C_D),
    ("C_f", C_f),
    ("D", D),
    ("L/D", C_L / C_D),
    ("Re", Re),
    ("W_w", W_w),
    ("W_w_strc", W_w_strc),
    ("W_w_surf", W_w_surf),
]

for name, sym in to_report:
    print(f"{name:12} = {sol(sym):.6g}")
