import numpy as np
import matplotlib.pyplot as plt


def gradient(x):
    # Compute the gradient of the function at point x
    return np.array([
        -np.exp(1 - x[0] - x[1]) + np.exp(x[0] + x[1] - 1) + 2 * x[0] + x[1] + 1,
        -np.exp(1 - x[0] - x[1]) + np.exp(x[0] + x[1] - 1) + x[0] + 2 * x[1] - 3
    ])


def hessian(x):
    # Compute the Hessian matrix of the function at point x
    exp1 = np.exp(1 - x[0] - x[1])
    exp2 = np.exp(x[0] + x[1] - 1)
    return np.array([
        [exp1 + exp2 + 2, exp1 + exp2 + 1],
        [exp1 + exp2 + 1, exp1 + exp2 + 2]
    ])


def armijo_backtracking(f, x, d, beta, t, sigma, f_x):
    alpha_k = t
    while f(x + alpha_k * d) > f_x + sigma * alpha_k * np.dot(gradient(x), d):
        alpha_k *= beta  # update the step size
    return alpha_k


# Common parameters
x = np.array([0, 0])  # initial point
epsilon = 1e-5  # stopping criteria
iter = 0

# Parameters for Armijo backtracking
sigma = 0.5  # Armijo condition parameter
beta = 0.5  # Backtracking factor
t = 1  # initial step size in line search

# Function definition
f = lambda x: np.exp(1 - x[0] - x[1]) + np.exp(x[0] + x[1] - 1) + x[0] ** 2 + x[0] * x[1] + x[1] ** 2 + x[0] - 3 * x[1]

# Plot the contour of cost function
x1_vals = np.arange(-2.5, 0.5, 0.01)
x2_vals = np.arange(0, 3, 0.01)
x1, x2 = np.meshgrid(x1_vals, x2_vals)
z = np.zeros_like(x1)

for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        z[i, j] = f([x1[i, j], x2[i, j]])

plt.figure(figsize=(8, 6))
plt.contour(x1, x2, z, levels=20)
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot of Cost Function')

# Newton's method
while np.linalg.norm(gradient(x)) > epsilon:
    # Form the search direction d using the gradient and Hessian
    H = hessian(x)
    g = gradient(x)

    try:
        d = -np.linalg.solve(H, g)  # Solve H * d = -g for d
    except :
        print("Hessian is singular or nearly singular. Stopping optimization.")
        break

    # Assign the step size using the Armijo backtracking function
    alpha_k = armijo_backtracking(f, x, d, beta, t, sigma, f(x))

    # Newton's update x to x_temp
    x_temp = x + alpha_k * d

    # Make plots
    plt.plot(x[0], x[1], '*r')
    plt.plot([x[0], x_temp[0]], [x[1], x_temp[1]], '-g')

    # Output the solution in each step
    iter = iter + 1
    x = x_temp

plt.savefig('./armjio_Newton.png')
# Print final information
print(
    f"Newton's meth0d' ends after {iter} iterations\nx*=[{x[0]:.6f},{x[1]:.6f}]\nf(x*)={f(x):.6f}\ngradient norm={np.linalg.norm(gradient(x)):.6e}")

plt.show()