import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(123)
data = np.array([[4, 15], [6, 8], [5, 17], [8, 15], [3, 1]])

x = np.arange(0, 10, 0.1)
y = 2 * x  + 1

plt.plot(x, y)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 10)
plt.yticks([0, 5, 10, 15, 20])
plt.show()
