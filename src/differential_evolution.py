from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from warnings import filterwarnings
from math import sqrt
from os import getcwd
from os import chdir
import matplotlib.pyplot as plt
import numpy as np


filterwarnings('ignore')


def ode_system(t, u, k1, k2, lambda_0, lambda_1):

    z2 = u[0]
    z3 = u[1]
    z4 = u[2]
    volume = u[3]
    z1 = z2 + z3 + z4

    cisplatin = 0
    psi = 20

    tgf = (lambda_0 * z1) / pow(pow(1 + lambda_0 * volume / lambda_1, psi), 1 / psi)

    dz1dt = tgf - k1 * cisplatin * z1
    dz2dt = k1 * cisplatin * z1 - k2 * z2
    dz3dt = k2 * z2 - k2 * z3
    dz4dt = k2 * z3 - k2 * z4
    dvdt = dz1dt + dz2dt + dz3dt + dz4dt

    return [dz2dt, dz3dt, dz4dt, dvdt]


def is_reference_time(times, ct):

    for t in times:
        if abs(ct - t) <= pow(10, -5):
            return True

    return False


def solve(x):

    global data, reference_times
    dt = 0.01
    final_t = 50
    times = np.arange(0, final_t + dt, dt)

    v, z2, z3, z4 = 500, 100, 200, 100
    u = [z2, z3, z4, v]

    k1 = x[0]
    k2 = x[1]
    lambda_0 = x[2]
    lambda_1 = x[3]
    params = (k1, k2, lambda_0, lambda_1)

    def solve_ode(t, y):
        return ode_system(t, y, *params)

    results = solve_ivp(solve_ode, (0, final_t), u, t_eval=times, method='Radau')
    u = results.y[:4, :]

    i, j = 0, 0
    v_error, v_sum = 0, 0

    for t in times:

        if is_reference_time(reference_times, t):

            v_data = data[i][4] + data[i][5] + data[i][6]
            v_error += (u[0][j] - v_data) * (u[0][j] - v_data)
            v_sum += v_data * v_data

            i += 1
        j += 1

    v_error = sqrt(v_error / v_sum)
    return v_error


def test_error(x, convergence):

    global error_list
    error_list.append(solve(x))


if __name__ == "__main__":

    chdir('..')

    global data, reference_times, error_list
    data = np.loadtxt(f'{getcwd()}/datasets/test.csv', delimiter=',')
    reference_times = data[:, 0]

    error_list = list()
    bounds = [
        (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1)
    ]

    solution = differential_evolution(solve, bounds,
                                      strategy='best1bin',
                                      maxiter=30,
                                      popsize=100,
                                      atol=pow(10, -5),
                                      tol=pow(10, -5),
                                      mutation=0.2,
                                      recombination=0.5,
                                      disp=True,
                                      workers=-1,
                                      callback=test_error)

    print(solution.x)
    print(solution.success)

    best = solution.x
    error = solve(best)

    print(f'Error: {error}')

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    ax.set(xlabel='Time', ylabel='Error', title='Error Evolution')
    ax.plot(range(len(error_list)), error_list)
    ax.grid()
    plt.show()
