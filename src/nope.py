import numpy as np
import matplotlib.pyplot as plt
import mlpy
np.random.seed(0)
x = np.arange(0, 2, 0.05).reshape(-1, 1) # training points
y = np.ravel(np.exp(x)) + np.random.normal(1, 0.2, x.shape[0]) # target values
xt = np.arange(0, 2, 0.01).reshape(-1, 1) # testing points
K = mlpy.kernel_gaussian(x, x, sigma=1) # training kernel matrix
Kt = mlpy.kernel_gaussian(xt, x, sigma=1) # testing kernel matrix
krr = mlpy.KernelRidge(lmb=0.01)
krr.learn(K, y)
yt = krr.pred(Kt)
fig = plt.figure(1)
plot1 = plt.plot(x[:, 0], y, 'o')
plot2 = plt.plot(xt[:, 0], yt)
plt.show()
