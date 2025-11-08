
# import all that good stuff
import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as real_numpy

# 1. Create an optimization problem instance (Opti object manages variables, constraints, and the solve process.)
opti = asb.Opti()

# 2. Define the optimization variables
# A more challenging starting point to see the solver's path, we can change this to anything
x = opti.variable(init_guess=-1.5)
y = opti.variable(init_guess=1.5)

# 3. Define the Rosenbrock function
# Our banana shaped function
f = (1 - x)**2 + 100 * (y - x**2)**2

# 4. Set the objective to minimize the function
# This tells our solver to find the minimum path
opti.minimize(f)

# 5. Set up a callback to record the optimization history
history = []
def callback(i):
    history.append((opti.debug.value(x), opti.debug.value(y)))

# 6. Solve the problem with the callback
# This saves the values in callback and solves this function
sol = opti.solve(callback=callback)

# 7. Print the results
print("Rosenbrock Function Minimum:")
print(f"  Optimal x: {sol.value(x):.4f}")
print(f"  Optimal y: {sol.value(y):.4f}")
print(f"  Minimum f(x, y): {sol.value(f):.4e}")


# 8. Create data for the plot
x_plot = real_numpy.linspace(-2, 2, 400)
y_plot = real_numpy.linspace(-1, 3, 400)
X, Y = real_numpy.meshgrid(x_plot, y_plot)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# 9. Create the plot
fig = plt.figure(figsize=(15, 7))

# 3D Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, rstride=100, cstride=100)
# Plot the path (this is the overlay)
history_x = [p[0] for p in history]
history_y = [p[1] for p in history]
history_z = (1 - real_numpy.array(history_x))**2 + 100 * (real_numpy.array(history_y) - real_numpy.array(history_x)**2)**2
ax1.plot(history_x, history_y, history_z, 'r.-', markersize=5, label='Solver Path')
ax1.plot(history_x[0], history_y[0], history_z[0], 'go', markersize=8, label='Start')
ax1.plot(sol.value(x), sol.value(y), sol.value(f), 'bo', markersize=8, label='Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('Rosenbrock Function Surface Plot')
ax1.legend()


# 2D Contour plot
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X, Y, Z, levels=real_numpy.logspace(0, 3.5, 15), cmap=cm.viridis)
ax2.plot(history_x, history_y, 'r.-', markersize=5, label='Solver Path')
ax2.plot(history_x[0], history_y[0], 'go', markersize=8, label='Start')
ax2.plot(sol.value(x), sol.value(y), 'bo', markersize=8, label='Minimum')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Rosenbrock Function Contour Plot')
ax2.legend()
plt.tight_layout()
plt.show()
