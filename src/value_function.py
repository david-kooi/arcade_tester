
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import numpy as np


def norm(peak_x, peak_y, mesh_x, mesh_y):
    C_squared = (peak_x - mesh_x)**2 + (peak_y - mesh_y)**2 

    return np.sqrt(C_squared)

def normalize(X, x_min, x_max):
    diff = x_max - x_min
    num  = X - x_min
    den  = np.max(X) - np.min(X)
    return diff*num/den + x_min 


scale  = 1
Ca     = 20
Cb     = 10
upper_limit = 20

peak_x = 5*scale
peak_y = 5*scale

xx = np.linspace(0,11*scale,11*scale)
yy = np.linspace(0,11*scale,11*scale)

X,Y = np.meshgrid(xx,yy)

D = norm(peak_x, peak_y, X, Y)  
D = D + 0.1

Z = Ca / (D**2 + Cb)
Z = np.clip(Z, 0, upper_limit)
#print(Z)

# Plot 3D
fig = plt.figure()
ax = fig.gca(projection="3d")
Z_3d = normalize(Z, 0, 1)
surf = ax.plot_surface(X,Y,Z_3d, cmap=cm.coolwarm, linewidth=0)
ax.set_zlim(-1, 2)
ax.set_xlabel("X")

# Plot 2D
fig_2d = plt.figure()
ax = fig_2d.gca()
Z_2d = normalize(Z, 0, 255)
plt.imshow(Z_2d, cmap='gray', vmin=0, vmax=255)
plt.show()





