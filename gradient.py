from math import *
import matplotlib.pyplot as plt
import time


def gradient(errorFunc, coefs, h=1e-05):
    grad = []
    for i, _ in enumerate(coefs):
        new_coefs = [x_j + (h if j == i else 0) for j, x_j in enumerate(coefs)]
        grad.append((errorFunc(new_coefs) - errorFunc(coefs)) / h)
    return grad


def generate_row():
    k = 8
    L = k / 100
    omega = 1000 / k
    dt = 2 * pi / 1000
    array = []
    array.append(0)
    array.append((-1) ** k * dt)
    for i in range(0, 1000 - 2):
        x = array[i + 1] * (2 + dt * L * (1 - array[i] ** 2)) - array[i] * (1 + dt ** 2 + dt * L * (1 - array[i] ** 2)) + dt ** 2 * sin(omega * dt)
        array.append(x)
    return array

def FurieApprox(a0, a1, b1, a2, b2, x):
    return a0/2.0 + a1*cos((x+500.0)*pi/500.0) + b1*sin((x+500.0)*pi/500.0) + a2*cos((2*(x+500.0))*pi/500.0) + b2*sin((2*(x+500.0))*pi/500.0)


def FurieError(coefficients):
    x = generate_row()
    a0 = coefficients[0]
    a1 = coefficients[1]
    b1 = coefficients[2]
    a2 = coefficients[3]
    b2 = coefficients[4]
    result_error = 0
    for i in range(1000):
        result_error += (FurieApprox(a0, a1, b1, a2, b2, i) - x[i]) ** 2
    return result_error


if __name__ == "__main__":
    values_y = generate_row()
    values_x = [i for i in range(len(values_y))]
    coefficients = [0,0,0,0,0]
    start = time.time_ns()
    stepsize = 0.01
    G = sqrt(sum(g_i ** 2 for g_i in gradient(FurieError, coefficients)))
    while abs(G) > 3:
        print(f"Gradient:{G}")
        coefficients = [x_i - (stepsize * dr / G) for x_i, dr in zip(coefficients, gradient(FurieError, coefficients))]
        G = sqrt(sum(g_i ** 2 for g_i in gradient(FurieError, coefficients)))
    end = time.time_ns()
    print('time:', (end - start) / 1000000000, 'sec')
    f2 = []
    for i in range(1000):
        f2.append(FurieApprox(coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], i))


    plt.plot(values_x, values_y, color='r', label='Data')
    plt.plot(values_x, f2, color='b', label='ML Data')
    plt.xlabel('Координата X', size=14)
    plt.ylabel('Координата Y', size=14)
    plt.legend()
    plt.show()
    print(coefficients)

