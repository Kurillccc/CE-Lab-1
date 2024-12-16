import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig

eps = 0.01

X01 = math.asin(-1/2) - 2*math.pi
X02 = -math.asin(-1/2) - 3*math.pi

def f1(gamma):
    def rhs(t, X):
        x, y = X
        return [y, -math.sin(x) - 1 / 2 - gamma * y]
    return rhs

################

def jacobian(gamma):
    return np.array([[0, 1], [-math.cos(X01), -gamma]])

def stability_analysis(gamma):
    J = jacobian(gamma)
    eigenvalues, eigenvectors = eig(J)

    return eigenvalues

def classify_equilibrium(eigenvalues):
    real_parts = np.real(eigenvalues)

    if np.all(real_parts < 0):
        return 'Устойчивое'
    elif np.all(real_parts > 0):
        return 'Неустойчивое'
    elif np.any(real_parts == 0):
        return 'Центр'
    else:
        return 'Седло'

##########################

def plot_points():
    plt. plot(0., 0., 'blue')

def eq_quiver(rhs, limits, N=16):
    xlims, ylims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            u, v = rhs(0.0, [x, y])
            U[i][j] = u/((u**2 + v**2)**(1/2))
            V[i][j] = v/((u**2 + v**2)**(1/2))
    return xs, ys, U, V

def plotonPlane(rhs, limits):
    plt.close()
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims [1])
    plt.ylim(ylims[0], ylims[1])
    xs, ys, U, V = eq_quiver(rhs, limits)
    plt.quiver(xs, ys, U, V, alpha=0.8)

# Ф. для нахождения с.р.
def findingStates0fEquilibrium():
    x01 = math.asin(-1 / 2) - 2 * math.pi
    x02 = -math.asin(-1 / 2) - 3 * math.pi
    xmins = [x01]
    xmaxs = [x02]
    while x01 <= 2*math.pi and x02 <= 2*math.pi:
        x01 += 2*math.pi
        x02 += 2*math.pi
        xmins.append(x01)
        xmaxs.append (x02)
    return xmins, xmaxs

def plotting(rhs, time, x, y, style):
    sol = solve_ivp(rhs, time, (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol.y
    plt.plot(x1, y1, style)

def drawingPortrait(rhs, gamma):
    xmins, xmaxs = findingStates0fEquilibrium()
    def drawingSeparatrices():
        v1 = -(gamma + math.sqrt(gamma ** 2 + 2 * math.sqrt(3))) / 2
        v2 = -(gamma - math.sqrt(gamma ** 2 + 2 * math.sqrt(3))) / 2
        for x in xmaxs:
            plotting(rhs, [0., -100.], x + eps, v1 * eps, 'y-')
            plotting(rhs, [0., 100.], x - eps, v1 * eps, 'y-')
            plotting(rhs, [0., 100.], x + eps, v2 * eps, 'y-')
            plotting(rhs, [0., -100.], x - eps, v2 *-eps, 'y-')
            plt.plot(x, 0., marker='x', color="red")
    if (gamma >= 0):
        for x in xmins:
            if (gamma < math.sqrt(2*math.sqrt(3))):
                plotting(rhs, [0., -100.], x, eps, 'r-')
                plotting(rhs, [0., -100.], x, 0.160, 'r-')
                plotting(rhs, [0., 100.], x, 0.160, 'r-')
                plt.plot(x, 0., marker='o', color="blue")
            else:
                plotting(rhs, [0., -100.], x + 1.5, 0, 'r-')
                plotting(rhs,[0., 100.], x + 1.5, 0, 'r-')
                plotting(rhs,[0., -100.], x + 2.5, -0.9, 'r-')
                plotting(rhs,[0., 100.], x + 2.5 , -0.9, 'r-')
                plotting(rhs, [0., -100.], x - 1, -0.5, 'r-')
                plotting(rhs, [0., 100.], x - 1, -0.5, 'r-')
                plotting(rhs, [0., -100.], x - 1.5, 0.5, 'r-')
                plotting(rhs, [0., 100.], x - 1.5, 0.5, 'r-')
                plt.plot(x, 0., marker='o', color="blue")
    else:
        for x in xmins:
            if (gamma > -math.sqrt(2*math.sqrt(3))):
                plotting(rhs, [0., 100.], x, eps, 'r-')
                plotting(rhs,[0., 100.], x, 1, 'r-')
                plotting(rhs, [0., -100.], x, 1, 'r-')
                plt.plot(x, 0., marker='x', color="red")
            else:
                plotting(rhs, [0., -100.], x + 1.5, 1, 'r-')
                plotting(rhs, [0., 100.], x + 1.5, 1, 'r-')
                plotting(rhs, [0., -100.], x + 2.5, -0.9, 'r-')
                plotting(rhs,[0., 100.], x + 2.5, -0.9,'r-')
                plotting(rhs,[0., -100.], x - 1, -0.5, 'r-')
                plotting(rhs, [0., 100], x - 1, -0.5, 'r-')
                plotting(rhs,[0., -100.], x - 1.5, 0.5, 'r-')
                plotting(rhs,[0., 100.], x - 1.5, 0.5, 'r-')
                plotting(rhs, [0., -100.], x + 0.25, 0.25, 'r-')
                plotting(rhs, [0., 100.], x + 0.25, 0.25,'r-')
                plt.plot(x, 0., marker='x', color="red")
    drawingSeparatrices()
    plt.ylabel("x'")
    plt.xlabel("x")
    # eigenvalues = stability_analysis(gamma)
    # stability = classify_equilibrium(eigenvalues)
    # plt.title(f'Stability: {stability}\ne1: {np.round(eigenvalues[0],4)} e2: {np.round(eigenvalues[1],4)}')

    plt.show()

def plotterXT(rhs, time, x, y, style):
    sol = solve_ivp(rhs, time, (x, y), method='RK45', rtol=1e-12)
    x, y = sol.y
    t = sol.t
    plt.plot(t, x, style)
    plt.ylabel("x(t)")
    plt.xlabel("t")

def plotterStableSep(rhs, time, x, y, style):
    sol = solve_ivp(rhs, time, (x, y), method='RK45', rtol=1e-6, atol=1e-12)
    x, y = sol.y
    t = sol.t
    for i, xi in enumerate(x):
        if xi < -math.asin(-1 / 2) - math.pi + eps*5 and i > 5:
            x = x[:i]
            t = t[:i]
            break
    plt.plot(t, x, style)
    plt.ylabel("x(t)")
    plt.xlabel("t")

def plotXT(rhs):
    plt.close()
    # по направлению св
    plt.subplot(2,3,1)
    plt.title("Финитные")
    plotterStableSep(rhs, [0., -20.], math.asin(-1/2), 0.8, 'g-')
    plt.subplot(2, 3, 2)
    plt.title("Устойчивая сепаратриса")
    plotterStableSep(rhs, [0., 1000.], -math.asin(-1 / 2) - math. pi, ((3 ** (1 / 4)) / (2 ** (1 / 2))), 'y-')
    plt. subplot(2, 3, 3)
    plt. title("Лимитационно-инфинитное")
    plotterStableSep(rhs, [0., 20.], -math.asin(-1 / 2) - math.pi, ((3 ** (1 / 4)) / (2 ** (1 / 2))), 'y-')
    plotterStableSep(rhs, [0., -20.], -math.asin(-1 / 2) - math.pi, ((3 ** (1 / 4)) / (2 ** (1 / 2))), 'y-')
    plt.subplot(2, 3, 4)
    plt. title( "Лимитационно-инфинитное")
    plotterStableSep(rhs, [0., 40.], -math.asin(-1 / 2) - math.pi, ((3 ** (1 / 4)) / (2 ** (1 / 2))), 'y-')
    plt. subplot(2, 3, 5)
    plt. title("Инфинитные")
    plotterStableSep(rhs, [0., 300.], math.asin(-1/2), 1.5, 'r-')
    plotterStableSep(rhs, [0., -200.], math.asin(-1/2), 1.5, 'r-')
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()

gamma_val = [0.1]
# [ 0, -6, 6, -1.47, 1.47, 1.86 ]

# -6 – неустойчивый узел и седло
# 6 – устойчивый узел и седло
# -1.47 – неустойчивый фокус и седло
# 1.47 – устойчивый фокус и седло
# 1.86 – вырожденный узел и седло

for i in range(2):
    rhs = f1(gamma_val[i])
    plotonPlane(rhs, [(-2 * math.pi,2 * math.pi), (-5.,5.)])
    drawingPortrait(rhs, i)
    plotXT(rhs)
