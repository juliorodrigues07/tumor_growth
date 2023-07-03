import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


delay = 0
steps = list()
check = False


def rk4(func, tk, _yk, _dt=0.01, **kwargs):

    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


def ode_system(t, y):

    global delay, check
    k1 = 0.52167446
    k2 = 0.01000157
    lambda_0 = 0.55265251
    lambda_1 = 0.99993298

    if delay == 2:
        check = True

    if check:
        z2 = steps[0]
        steps.pop(0)
        delay = 1
    else:
        z2 = 0

    z1 = y[0]
    steps.append(y[1])
    volume = y[2]

    cisplatin = 0 if delay == 0 else 0
    psi = 20

    # Tumor growth function
    tgf = (lambda_0 * z1) / pow(1 + pow(lambda_0 * volume / lambda_1, psi), 1 / psi)

    dz1dt = tgf - k1 * cisplatin * z1
    dz2dt = k1 * cisplatin * z1 - k2 * z2
    dvdt = dz1dt + dz2dt

    delay += 1
    return np.array([dz1dt, dz2dt, dvdt])


def simulate():

    global delay, check

    # Time step and ending point
    dt = 0.01
    tfinal = 25
    time = np.arange(0, tfinal + dt, dt)

    # Initial conditions
    y0 = np.array([82, 50, 132])

    yk = y0
    state_history = list()

    # Simulation (time series loop)
    t = 0
    for t in time:
        state_history.append(yk)
        yk = rk4(ode_system, t, yk, dt)

        delay = 0
        check = False
        steps.clear()

    state_history = np.array(state_history)
    print(f'y evaluated at time t = {t} seconds: {yk[0]}')

    return state_history, time


def exp_and_plot(data, time):

    df = pd.DataFrame([row[2] for row in data], columns=['Volume'])
    df.insert(0, 'Time', time)
    df.to_csv('results.csv', float_format='%.5f', sep=',')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.plot(time, data[:, 2], label='Volume', color='green')

    ax.set(xlabel='Time (Days)', ylabel='Volume', title='Tumor Growth (RK4)')
    ax.grid()
    fig.savefig('test.svg', format='svg')

    plt.legend(['Volume'], loc='best')
    plt.show()


def main():
    data, time = simulate()
    exp_and_plot(data, time)


if __name__ == '__main__':
    main()
