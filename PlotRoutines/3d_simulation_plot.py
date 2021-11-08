# -*- coding: utf-8 -*-
"""
Visualization of a 3D structure defined inside a sim.Simulation instance

See also
--------
Meep Tutorial: Visualizing 3D structures
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
#from mayavi import mlab
import math

#%% DEFAULT SIMULATION
# Do not run this if you have your own simulation defined

cell_size = mp.Vector3(2,2,2)

# A hexagon is defined as a prism with six vertices centered on the origin
vertices = [mp.Vector3(-1,0),
            mp.Vector3(-0.5,math.sqrt(3)/2),
            mp.Vector3(0.5,math.sqrt(3)/2),
            mp.Vector3(1,0),
            mp.Vector3(0.5,-math.sqrt(3)/2),
            mp.Vector3(-0.5,-math.sqrt(3)/2)]

geometry = [mp.Prism(vertices, height=1.0, material=mp.Medium(index=3.5)),
            mp.Cone(radius=1.0, radius2=0.1, height=2.0, material=mp.air)]

sim = mp.Simulation(resolution=50,
                    cell_size=cell_size,
                    geometry=geometry)

#%% STEPS TO GET DATA FROM SIMULATION

sim.init_sim()

x, y, z, *more = sim.get_array_metadata() # (x,y,z,w) = sim.get_array_metadata()
# Returns coordinates and interpolation weights of the fields :)
del more

eps_data = sim.get_epsilon()

#%% 3D VISUALIZATION VIA MAYABI SUGGESTED BY MEEP

#s = mlab.contour3d(eps_data, colormap="YlGnBu")
#mlab.show()

#%% 3D SLICED VISUALIZATION VIA MATPLOTLIB

n_slices = 3

x0 = x[int(len(x)/2)]
y0 = y[int(len(y)/2)]
z0 = z[int(len(z)/2)]

i_x0 = int(eps_data.shape[0]/2)
i_y0 = int(eps_data.shape[1]/2)
i_z0 = int(eps_data.shape[2]/2)

titles = ["X=X0", "Y=Y0", "Z=Z0"]
eps_slices = [eps_data[i_x0, :, :], eps_data[:, i_y0, :], eps_data[:, :, i_z0]]

x_plot = [y, x, x]
x_labels = ["Y", "X", "X"]
y_plot = [z, z, y]
y_labels = ["Z", "Z", "Y"]

cmlims = (np.min(eps_slices), np.max(eps_slices))
cmlims = (cmlims[0]-.1*(cmlims[1]-cmlims[0]),
         cmlims[1]+.1*(cmlims[1]-cmlims[0]))

nplots = n_slices
fig, axes = plt.subplots(1, n_slices)
fig.subplots_adjust(hspace=0.5, wspace=.5)
fig.set_figheight = 6.4
fig.set_figwidth = n_slices * 6.4
plt.show()

for ax, tl in zip(axes, titles):
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(tl)

for ax, xp, yp, eps, tl, xl, yl in zip(axes, x_plot, y_plot, eps_slices, 
                                       titles, x_labels, y_labels):
    ax.imshow(eps.T, interpolation='spline36', cmap='RdBu', 
              vmin=cmlims[0], vmax=cmlims[1])
    ax.axis("off")
    ax.xaxis.set_label_text(xl)
    ax.yaxis.set_label_text(yl)
