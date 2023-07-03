from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from warnings import filterwarnings
from pandas import read_csv
from math import sqrt
from os import getcwd
from os import chdir
import matplotlib.pyplot as plt
import numpy as np


filterwarnings('ignore')
delay = 0
check = False
steps = list()


def ode_system(t, u, k1, k2, lambda_0, lambda_1):

    global delay, check, cisplatin

    if delay == 2:
        check = True

    if check:
        z2 = steps[0]
        steps.pop(0)
        delay = 1
    else:
        z2 = 0

    z1 = u[0]
    steps.append(u[1])
    volume = u[2]

    # Tumor growth function
    psi = 20
    tgf = (lambda_0 * z1) / pow(1 + pow(lambda_0 * volume / lambda_1, psi), 1 / psi)

    dz1dt = tgf - k1 * cisplatin * z1
    dz2dt = k1 * cisplatin * z1 - k2 * z2
    dvdt = abs(dz1dt) + abs(dz2dt)

    delay += 1
    return [dz1dt, dz2dt, dvdt]


def is_reference_time(times, ct):

    for t in times:
        if abs(ct - t) <= pow(10, -5):
            return True

    return False


def solve(x):

    global data, reference_times, delay, check
    dt = 0.01
    final_t = 50
    times = np.arange(0, final_t + dt, dt)

    z1, z2 = 82, 50
    v = z1 + z2
    u = [z1, z2, v]

    k1 = x[0]
    k2 = x[1]
    lambda_0 = x[2]
    lambda_1 = x[3]
    params = (k1, k2, lambda_0, lambda_1)

    def solve_ode(t, y):
        return ode_system(t, y, *params)

    results = solve_ivp(solve_ode, (0, final_t), u, t_eval=times, method='Radau')
    u = results.y[:3, :]

    i, j = 0, 0
    z1_error, z2_error, v_error = 0, 0, 0
    z1_sum, z2_sum, v_sum = 0, 0, 0

    for t in times:

        if is_reference_time(reference_times, t):

            z1_data = data[i][5]
            z2_data = data[i][4]
            v_data = data[i][6]

            z1_error += (u[0][j] - z1_data) * (u[0][j] - z1_data)
            z2_error += (u[1][j] - z2_data) * (u[1][j] - z2_data)
            v_error += (u[2][j] - v_data) * (u[2][j] - v_data)

            z1_sum += z1_data * z1_data
            z2_sum += z2_data * z2_data
            v_sum += v_data * v_data

            i += 1
        j += 1

    delay = 0
    check = False
    steps.clear()

    z1_error = sqrt(z1_error / z1_sum)
    z2_error = sqrt(z2_error / z2_sum)
    v_error = sqrt(v_error / v_sum)

    return z1_error + z2_error + v_error


def test_error(x, convergence):

    global error_list
    error_list.append(solve(x))


if __name__ == "__main__":

    chdir('..')

    global data, reference_times, error_list, cisplatin

    untreated_rats, solutions = list(), list()
    dataset = read_csv(f'{getcwd()}/datasets/control_22aug.csv')
    n_rats = dataset['ID'].max()

    for i in range(1, n_rats + 1):
        grouped = dataset.groupby(dataset['ID'])
        rat = np.array(grouped.get_group(i))
        untreated_rats.append(rat)

    if n_rats == 21:
        cisplatin = 0
        print('ADJUSTING UNTREATED RATS DATASET\n')
    else:
        cisplatin = 5
        print('ADJUSTING DAY 0 UNIQUE CISPLATIN DOSE TREATED RATS DATASET\n')

    for i in range(n_rats):

        print(f'\nFITTING RAT {i + 1} DATA:\n')
        data = untreated_rats[i]
        reference_times = np.array([row[0] for row in untreated_rats[i]])

        error_list = list()
        bounds = [
            (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1)
        ]

        solution = differential_evolution(solve, bounds,
                                          strategy='best1bin',
                                          maxiter=30,
                                          popsize=100,
                                          atol=pow(10, -3),
                                          tol=pow(10, -3),
                                          mutation=0.2,
                                          recombination=0.5,
                                          disp=True,
                                          workers=6,
                                          callback=test_error)

        solutions.append(solution.x)
        print(f'Parameters Solution: {solution.x}')
        print(f'Converged: {"Yes" if solution.success else "No"}')

        best = solution.x
        error = solve(best)
        print(f'Associated Error: {error}\n')

    print(f'\nFinal Parameters: {np.mean(solutions, axis=0)}')

    # fig, ax = plt.subplots()
    # fig.set_size_inches(12, 8)
    #
    # ax.set(xlabel='Time', ylabel='Error', title='Error Evolution')
    # ax.plot(range(len(error_list)), error_list)
    # ax.grid()
    # plt.show()
