import numpy as np
import matplotlib.pyplot as plt


def armijo(f, x, d, beta, t, sigma, f_x):
    alpha = t
    while f(x + alpha * d) > f_x + sigma * alpha * np.dot(gradient(x), d):
        alpha *= beta  # update the step size
    return alpha


def gradient(x):
    # Compute the gradient of the function at point x
    return np.array([
        -np.exp(1 - x[0] - x[1]) + np.exp(x[0] + x[1] - 1) + 2 * x[0] + x[1] + 1,
        -np.exp(1 - x[0] - x[1]) + np.exp(x[0] + x[1] - 1) + x[0] + 2 * x[1] - 3
    ])


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

# User input for strategy selection
s = int(input(
    'Which way to use?\n [1]: stepsize = 1\n [2]: stepsize = 0.1\n [3]: armijo/backtracking\n'))

# Plot the contour of cost function
x1, x2 = np.meshgrid(np.arange(-2.5, 0.5, 0.01), np.arange(0, 3, 0.01))
z = np.zeros_like(x1)
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        z[i, j] = f([x1[i, j], x2[i, j]])

plt.figure(1)
plt.contour(x1, x2, z, 20)
plt.colorbar()

# Gradient descent algorithm
while np.linalg.norm(gradient(x)) > epsilon:
    d = -gradient(x)  # Search direction is the negative of the gradient

    if s == 1:
        alpha_k = 1  # Large constant step size
    elif s == 2:
        alpha_k = 0.1  # Small constant step size
    else:
        alpha_k = armijo(f, x, d, beta, t, sigma, f(x))

    x_temp = x + alpha_k * d  # GD update x to x_temp

    # Make some plots
    plt.plot(x[0], x[1], '*r')
    plt.plot([x[0], x_temp[0]], [x[1], x_temp[1]], '-g')

    # Output the solution in each step
    iter = iter + 1
    x = x_temp

if s == 1:
    plt.savefig('./large_constant_gd.png')
elif s == 2:
    plt.savefig('./small_constant_gd.png')
else:
    plt.savefig('./armjio_gd.png')
# Print final information
print(
    f'The algorithm ends after {iter} iterations\nx*=[{x[0]:.6f},{x[1]:.6f}]\nf(x*)={f(x):.6f}\ngradient norm={np.linalg.norm(gradient(x)):.6e}')

plt.show()