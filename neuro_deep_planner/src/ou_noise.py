import numpy as np
import numpy.random as nr

# MU = Center value, THETA = How strong is noise pulled to MU, SIGMA = Variance of noise)


class OUNoise:

    def __init__(self, action_dimension, mu=0, theta=0.5, sigma=0.1):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    ou = OUNoise(2, 0, 0.1, 0.1)
    states = []
    for i in range(100):
        if i == 20:
            ou.reset()
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
