import numpy as np
import matplotlib.pyplot as plt

DIR_IMAGES = 'images'
E = -65
R = 10
τ = 10
VTH = -50
V0 = -65
TMAX, ΔT = 200, .05

dvdt = lambda t, v, I: integrate_and_fire(t, v, I, E, R, τ)
v = lambda t, I: v_t(t, I, E, R, τ, V0)
ω = lambda I: frequency(I, E, R, τ, VTH, V0)

def integrate_and_fire(t, v, I, E, R, τ):
    return (E - v + R * I(t))/τ

def v_t(t, I, E, R, τ, v0):
    s = R * I(t) + E
    return np.exp(-t/τ) * (v0 - s) + s

def frequency(I, E, R, τ, vth, v0):
    if I >= (vth - E)/R:
        T = -τ * np.log(1 + (E - vth)/(R * I))
        return 1/T
    else:
        return 0

class RungeKutta4:
    def __init__(self, F, x0, vth, tf, Δt):
        self.F = F
        self.x0 = x0
        self.vth = vth
        self.tf = tf
        self.Δt = Δt
    
    def __step(self, tn, xn):
        k1 = self.F(tn, xn)*self.Δt
        k2 = self.F(tn, xn + .5*k1)*self.Δt
        k3 = self.F(tn, xn + .5*k2)*self.Δt
        k4 = self.F(tn, xn + k3)*self.Δt
        return xn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    
    def run(self):
        xn = self.x0
        tn = 0
        ω = 0
        xs = []
        while tn <= self.tf:
            xn = self.__step(tn, xn)
            if(xn >= self.vth):
                xn = self.x0
                ω += 1
            xs.append(xn)
            tn += self.Δt
        return ω/self.tf, xs

def point_a_b():
    I = lambda t: 2
    F = lambda t, v: dvdt(t, v, I)
    rk4 = RungeKutta4(F, E, VTH, TMAX, ΔT)

    t = np.arange(0, TMAX, ΔT, dtype='float')
    _, v_aprox = rk4.run()

    plot_VvsT(t, v_aprox, v(t, I))

def point_c():
    N = 7
    ω_aprox = []
    for current in range(N):
        I = lambda t: current
        F = lambda t, v: dvdt(t, v, I)
        rk4 = RungeKutta4(F, E, VTH, TMAX, ΔT)
        ωrk, _ = rk4.run()
        ω_aprox.append(ωrk)

    Is = np.arange(0, N, 0.1, dtype='float')
    plot_ωvsI(Is, ω_aprox, [ω(I) for I in Is])

def point_d():
    C = lambda t: np.cos(t/3) + np.cos(t/7) + np.cos(t/13)
    S = lambda t: np.sin(t/5) + np.sin(t/11)
    I = lambda t: .35 * (S(t) + C(t))**2
    F = lambda t, v: dvdt(t, v, I)

    t = np.arange(0, TMAX, ΔT, dtype='float')
    rk4 = RungeKutta4(F, E, VTH, TMAX, ΔT)
    _, v_aprox = rk4.run()

    plot_VvsT_varyingI(t, v_aprox, VTH, I(t))

def plot_VvsT(t, v_aprox, v, save=True):
    plt.xlabel('Tiempo - t[ms]', fontsize=13)
    plt.ylabel('Potencial de Membrana - [mV]', fontsize=13)

    plt.plot(t, v_aprox, '-g', label='Aprox. Numérica')
    plt.plot(t, v, '-b', label='Sol. Analítica')

    plt.legend(framealpha=1, frameon=True, loc='upper right')
    plt.grid()

    plt.savefig(f'{DIR_IMAGES}/VvsT.png') if save else plt.show()
    plt.clf()

def plot_ωvsI(I, ω_aprox, ω, save=True):
    plt.xlabel('Corriente Externa I - [nA]', fontsize=13)
    plt.ylabel('Frecuencia ω - [1/ms]', fontsize=13)
    
    plt.plot(ω_aprox, 'ro', label='Aproximación')
    plt.plot(I, ω, '-b', label='ω(I)')

    plt.legend(framealpha=1, frameon=True, loc='lower right')
    plt.grid()

    plt.savefig(f'{DIR_IMAGES}/frequency.png') if save else plt.show()
    plt.clf()

def plot_VvsT_varyingI(t, v_aprox, vth, I, save=True):

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_ylabel('Potencial de Mem. - [mV]', fontsize=10)
    ax2.set_ylabel('Corriente - [nA]', fontsize=10)
    ax2.set_xlabel('Tiempo - t[ms]', fontsize=10)

    horiz_line_vth = np.array([vth for i in range(len(t))])
    ax1.plot(t, horiz_line_vth, '--r', label='Umbral')
    ax1.plot(t, v_aprox, '-b', label=r'$V_m(t)$ c/d')
    ax2.plot(t, I, '-g', label=r'$I_e(t)$')

    ax1.legend(framealpha=1, frameon=True, loc='upper right')
    ax1.grid()
    ax2.legend(framealpha=1, frameon=True, loc='upper right')
    ax2.grid()

    plt.savefig(f'{DIR_IMAGES}/VvsT_varyingI.png') if save else plt.show()
    plt.clf()

def main():
    point_a_b()
    point_c()
    point_d()
    

if __name__ == "__main__":
    main()