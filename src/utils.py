TIMING = 2
PSI = 20


def analyze_delay(time):

    # Controlling delay application (must be at least N steps ahead in time {t = 0})
    if time == TIMING:
        return True
    else:
        return False


def calculate_tgf(lambda_0, lambda_1, z1, v):

    # Tumor growth function (psi defined based on article definitions)
    return (lambda_0 * z1) / pow(1 + pow(lambda_0 * v / lambda_1, PSI), 1 / PSI)
