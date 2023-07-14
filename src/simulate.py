from matplotlib.backends.backend_pdf import PdfPages
from warnings import filterwarnings
from utils import analyze_delay
from utils import calculate_tgf
from pandas import read_csv
from os.path import isdir
from os import getcwd
from os import mkdir
from os import chdir
import numpy as np
import matplotlib.pyplot as plt


filterwarnings('ignore')
delay = 0
steps = list()
check = False

chdir('..')
if not isdir(f'{getcwd()}/plots'):
    mkdir(f'{getcwd()}/plots')

pdf = PdfPages(f'{getcwd()}/plots/untreated.pdf')


def rk4(func, tk, _yk, _dt=0.01, **kwargs):

    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


def ode_system(t, y):

    global delay, check

    # DE estimated parameters for untreated and treated rats
    k1 = 0.52167446
    k2 = 0.01000157
    lambda_0 = 0.55265251
    lambda_1 = 0.99993298

    # k1 = 0.02867228
    # k2 = 0.1863795
    # lambda_0 = 0.55193695
    # lambda_1 = 0.8883638

    # Maintains a list with the 2 previous values from Z2 compartiment, apllying delay to the ODE
    check = analyze_delay(delay)
    if check:
        z2 = steps[0]
        steps.pop(0)
        delay = 1
    else:
        z2 = 0

    # Tumor volume values
    z1 = y[0]
    steps.append(y[1])
    volume = y[2]

    # Medication applying
    cisplatin = 0
    # cisplatin = 5 if delay == 0 else 0

    tgf = calculate_tgf(lambda_0, lambda_1, z1, volume)

    # ODEs: Z1 and Z2 compartment, and total tumor volume growth in time
    dz1dt = tgf - k1 * cisplatin * z1
    dz2dt = k1 * cisplatin * z1 - k2 * z2
    dvdt = dz1dt + dz2dt

    delay += 1
    return np.array([dz1dt, dz2dt, dvdt])


def simulate(initial_condition, t_final):

    global delay, check

    # Time step and ending point
    dt = 0.01
    time = np.arange(0, t_final + dt, dt)

    # Initial conditions
    y0 = np.array(initial_condition)

    yk = y0
    state_history = list()

    # Simulation (time series loop)
    t = 0
    for t in time:
        state_history.append(yk)
        yk = rk4(ode_system, t, yk, dt)

        # After each simulation, restarts delay, emptying the list containing z2 quantities N steps back
        delay = 0
        check = False
        steps.clear()

    state_history = np.array(state_history)
    return state_history, time


def exp_and_plot(data, time, rat, index):

    # df = pd.DataFrame([row[2] for row in data], columns=['Volume'])
    # df.insert(0, 'Time', time)
    # df.to_csv(f'{getcwd()}/plots/results.csv', float_format='%.5f', sep=',')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.plot(time, data[:, 2], label='Simulated Volume', color='red')

    days = [i + 1 for i in range(len(rat))]
    ax.plot(days, [row[6] for row in rat], marker='o', label='Experimental Data', color='blue')

    ax.set(xlabel='Time (Days)', ylabel='Volume', title=f'Tumor Growth - Rat {index + 1} (RK4)')
    ax.grid()
    plt.legend(['Simulated Volume', 'Experimental Data'], loc='best')

    pdf.savefig(fig)
    # plt.show()


def main():

    rats = list()

    # dataset = read_csv(f'{getcwd()}/datasets/cisplat_19sep.csv')
    dataset = read_csv(f'{getcwd()}/datasets/control_22aug.csv')
    n_rats = dataset['ID'].max()

    # Splits the dataset by ID (a list containing matrixes from each rat data)
    for i in range(1, n_rats + 1):
        grouped = dataset.groupby(dataset['ID'])
        rat = np.array(grouped.get_group(i))
        rats.append(rat)

    for i in range(n_rats):

        volume = rats[i][0][6]
        z1 = rats[i][0][1]
        z2 = rats[i][0][2]

        # Number of observations (in days) for each rat
        t_final = len(rats[i])
        initial_condition = [z1, z2, volume]

        data, time = simulate(initial_condition, t_final)
        exp_and_plot(data, time, rats[i], i)

    pdf.close()


if __name__ == '__main__':
    main()
