from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from warnings import filterwarnings
from utils import analyze_delay
from utils import calculate_tgf
from pandas import read_csv
from os.path import isdir
from math import sqrt
from os import getcwd
from os import mkdir
from os import chdir
import matplotlib.pyplot as plt
import numpy as np


filterwarnings('ignore')
delay = 0
check = False
steps = list()

chdir('..')
if not isdir(f'{getcwd()}/plots'):
    mkdir(f'{getcwd()}/plots')


def ode_system(t, u, k1, k2, lambda_0, lambda_1):

    global delay, check, cisplatin

    # Maintains a list with the 2 previous values from Z2 compartiment, apllying delay to the ODE
    check = analyze_delay(delay)
    if check:
        z2 = steps[0]
        steps.pop(0)
        delay = 1
    else:
        z2 = 0

    # Tumor volume values
    z1 = u[0]
    steps.append(u[1])
    volume = u[2]

    tgf = calculate_tgf(lambda_0, lambda_1, z1, volume)

    if delay != 0:
        cisplatin = 0

    # ODEs: Z1 and Z2 compartment, and total tumor volume growth in time
    dz1dt = tgf - k1 * cisplatin * z1
    dz2dt = k1 * cisplatin * z1 - k2 * z2
    dvdt = dz1dt + dz2dt

    delay += 1
    return [dz1dt, dz2dt, dvdt]


def is_reference_time(times, ct):

    for t in times:
        if abs(ct - t) <= pow(10, -5):
            return True

    return False


def solve(x):

    global data, reference_times, delay, check

    # Time step and ending point
    dt = 0.01
    final_t = 50
    times = np.arange(0, final_t + dt, dt)

    # Initial conditions
    v = data[0][6]
    z1, z2 = data[0][1], data[0][2]
    u = [z1, z2, v]

    # Parameters for estimation
    k1 = x[0]
    k2 = x[1]
    lambda_0 = x[2]
    lambda_1 = x[3]
    params = (k1, k2, lambda_0, lambda_1)

    def solve_ode(t, y):
        return ode_system(t, y, *params)

    # Simulation (time series loop)
    results = solve_ivp(solve_ode, (0, final_t), u, t_eval=times, method='Radau')
    u = results.y[:3, :]

    i, j = 0, 0
    z1_error, z2_error, v_error = 0, 0, 0
    z1_sum, z2_sum, v_sum = 0, 0, 0

    for t in times:

        if is_reference_time(reference_times, t):

            z1_data = data[i][1]
            z2_data = data[i][2]
            v_data = data[i][6]

            z1_error += (u[0][j] - z1_data) * (u[0][j] - z1_data)
            z2_error += (u[1][j] - z2_data) * (u[1][j] - z2_data)
            v_error += (u[2][j] - v_data) * (u[2][j] - v_data)

            z1_sum += z1_data * z1_data
            z2_sum += z2_data * z2_data
            v_sum += v_data * v_data

            i += 1
        j += 1

    # After each simulation, restarts delay, emptying the list containing z2 quantities N steps back
    delay = 0
    check = False
    steps.clear()

    # Norm 2 errors
    z1_error = sqrt(z1_error / z1_sum)
    z2_error = sqrt(z2_error / z2_sum)
    v_error = sqrt(v_error / v_sum)

    return z1_error + z2_error + v_error


def test_error(x, convergence):

    global error_list
    error_list.append(solve(x))


def main():

    global data, reference_times, error_list, cisplatin
    rats, solutions = list(), list()

    # dataset = read_csv(f'{getcwd()}/datasets/cisplat_19sep.csv')
    dataset = read_csv(f'{getcwd()}/datasets/control_22aug.csv')
    n_rats = dataset['ID'].max()

    # Splits the dataset by ID (a list containing matrixes from each rat data)
    grouped = dataset.groupby(dataset['ID'])
    for i in range(1, n_rats + 1):
        rat = np.array(grouped.get_group(i))
        rats.append(rat)

    # 40 rats ==> 19 treated | 21 untreated
    if n_rats == 21:
        cisplatin = 0
        print('ADJUSTING UNTREATED RATS DATASET\n')
    else:
        cisplatin = 5
        print('ADJUSTING DAY 0 UNIQUE CISPLATIN DOSE TREATED RATS DATASET\n')

    for i in range(n_rats):

        error_list = list()
        print(f'\nFITTING RAT {i + 1} DATA:\n')

        # First column has the observation times (in days)
        data = rats[i]
        reference_times = np.array([row[0] for row in rats[i]])

        # Limits for adjusting parameters (k1, k2, lambda_0, lambda_1)
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

        # fig, ax = plt.subplots()
        # fig.set_size_inches(12, 8)

        # ax.set(xlabel='Time', ylabel='Error', title=f'Error Evolution - Rat {i + 1}')
        # ax.plot(range(len(error_list)), error_list)
        # ax.grid()

        # fig.savefig(f'{getcwd()}/plots/error_rat{i + 1}_untreated.svg', format='svg')
        # plt.show()

    # Mean between all the parameter adjusting done for each rat
    print(f'\nFinal Parameters: {np.mean(solutions, axis=0)}')


if __name__ == "__main__":
    global data, reference_times, error_list, cisplatin
    main()
