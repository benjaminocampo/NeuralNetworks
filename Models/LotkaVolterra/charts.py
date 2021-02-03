import numpy as np
import matplotlib.pyplot as plt

DIR_IMAGES = 'images'

class RungeKutta4:
    def __init__(self, M, X⃗0, ts, Δt):
        self.M = M
        self.X⃗0 = X⃗0
        self.ts = ts
        self.Δt = Δt
    
    def __step(self, X⃗n):
        K⃗1 = self.M.F⃗(X⃗n)*self.Δt
        K⃗2 = self.M.F⃗(X⃗n + .5*K⃗1)*self.Δt
        K⃗3 = self.M.F⃗(X⃗n + .5*K⃗2)*self.Δt
        K⃗4 = self.M.F⃗(X⃗n + K⃗3)*self.Δt
        return X⃗n + 1/6*(K⃗1 + 2*K⃗2 + 2*K⃗3 + K⃗4)
    
    def run(self):
        X⃗n = self.X⃗0
        xs = []
        for _ in self.ts:
            X⃗n = self.__step(X⃗n)
            xs.append(X⃗n)
        return xs


class LotkaVolterraModeler:
    def __init__(self, α=.1, β=.02, γ=.3, δ=.01):
        self.α = α
        self.β = β
        self.γ = γ
        self.δ = δ

    def Ċ(self, X⃗):
        c, z = X⃗
        return self.α*c - self.β*c*z

    def Ż(self, X⃗):
        c, z = X⃗
        return -self.γ*z + self.δ*c*z

    def F⃗(self, X⃗):
        return np.array([self.Ċ(X⃗), self.Ż(X⃗)])

    def plot_phase_portrait(self, width, heigth, save=False):
        x = np.linspace(-10, width)
        y = np.linspace(-10, heigth)

        X⃗ = np.meshgrid(x, y)
        c, z = X⃗

        plt.streamplot(c, z, self.Ċ(X⃗), self.Ż(X⃗), density=1)
        plt.plot(0, 0,'ro')        
        plt.plot(self.γ/self.δ, self.α/self.β,'ro')

        plt.xlabel('Población de Conejos - C(t)', fontsize=13)
        plt.ylabel('Población de Zorros - Z(t)', fontsize=13)
        plt.grid()

        plt.savefig(f'{DIR_IMAGES}/flow_diagram.png') if save else plt.show()
        plt.clf()

    def plot_CvsZ(self, xs, save=False):
        cs, zs = zip(*xs)

        plt.plot(cs, zs, 'b')

        plt.xlabel('Población de Conejos - C(t)', fontsize=13)
        plt.ylabel('Población de Zorros - Z(t)', fontsize=13)

        plt.grid()

        plt.savefig(f'{DIR_IMAGES}/CvsZ.png') if save else plt.show()
        plt.clf()

    def plot_CZvsT(self, ts, xs, save=False):
        cs, zs = zip(*xs)
        
        plt.xlabel('Tiempo - t', fontsize=13)
        plt.ylabel('Población', fontsize=13)
        
        plt.plot(ts, cs, '-b', label='C(t)')
        plt.plot(ts, zs, '-r', label='Z(t)')

        plt.legend(framealpha=1, frameon=True)
        plt.grid()

        plt.savefig(f'{DIR_IMAGES}/CZvsT.png') if save else plt.show()
        plt.clf()


def main():
    ti, tf, Δt = 0, 200, .05
    X⃗0, ts = np.array([40, 9]), np.arange(ti, tf, Δt, dtype='float')

    M = LotkaVolterraModeler()
    rk4 = RungeKutta4(M, X⃗0, ts, Δt)

    xs = rk4.run()
    M.plot_CvsZ(xs, save=True)
    M.plot_CZvsT(ts, xs, save=True)

    width, heigth = 50, 15
    M.plot_phase_portrait(width, heigth, save=True)


if __name__ == "__main__":
    main()
