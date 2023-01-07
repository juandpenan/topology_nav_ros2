import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import pathlib
import torch
from transformers import RobertaModel,RobertaTokenizerFast
from itertools import repeat

from multiprocessing import Pool

def f(x,y):
    return x*x,x+x


if __name__ == '__main__':
    with Pool(5) as p:
        x,y = zip(*p.starmap(f, [(1,2), (2,3), (3,4)]))
    print(list(x))

# """Localize a mobile robot in a cyclic grid by probabilistic Markov localization.

# https://www.cs.princeton.edu/courses/archive/fall11/cos495/COS495-Lecture14-MarkovLocalization.pdf
# """

# import numpy as np
# import scipy.stats as st
# from scipy import ndimage
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt


# # ind = 364437
# ans = ['no', 'no', 'cup', 'white', 'no', 'no', 'no', 'yes', 'yes', 'no']
# conf = [0.8,0.5,0.6 ,0.7 ,0.8,0.1,0.45,0.78,0.8,0.9]

# a = np.load("/home/juan/Workspaces/phd_ws/src/topological_localization/topological_mapping/map/map.npy",allow_pickle=True)
# # a = np.zeros(shape=(231,383,8,10,2),dtype=object)
# grid = np.zeros(shape=(231,383,9))
# # ind_arr = np.arange(707784).reshape(231,383,8)
# # y,x,state = np.where(ind_arr == ind)

# # for i in range(len(ans)):
# #     a[y,x,state,i] = ans[i],conf[i]
# # print(a[122,247,0])
# indexs = []
# # x:192.000000 y:117.000000 
# #creo que esta trucado
# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         for s in range(a.shape[2]):
#             sum = 0
#             for x in range(10):
#                 if a[i,j,s,x][0] == ans[x]:
#                     # indexs.append([i,j,s])
#                     sum += conf[x]          
#                     grid[i,j,s+1] = sum
# grid = grid / np.sum(grid)
# print (np.unravel_index(np.argmax(grid, axis=None), grid.shape))

# np.save("array.npy",a) 118 360 

    

# def gauss_1d_kernel(k_size=5,sigma=1.0,center=2):
    

#     # Define the kernel size
#     n = k_size

#     # Create an empty kernel
#     kernel = np.zeros(n)

#     # Calculate the values of the Gaussian distribution at each element of the kernel
#     for x in range(n):
#         kernel[x] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - center)**2 / (2 * sigma**2)))

#     # Normalize the kernel so that the values sum to 1
#     kernel = kernel / np.sum(kernel)

#     return kernel

# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel."""

#     x = np.linspace(-nsig, nsig, kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kern2d = np.outer(kern1d, kern1d)
#     return kern2d/kern2d.sum() 
# def gaussian_2d_kernel(center = (2,2),k_size = (10,10),sig=[1,1]):


#     # Define the kernel size
#     n_h = k_size[0]
#     n_w = k_size[1]

#     # Define the standard deviation of the Gaussian distribution for each axis
#     sigma_h = sig[0]
#     sigma_w = sig[1]

#     # Define the center of the kernel
#     center_h = center[0]
#     center_w = center[1]


#     # Create an empty kernel
#     kernel = np.zeros((n_h, n_w))

#     # Calculate the values of the Gaussian distribution at each element of the kernel
#     for h in range(n_h):
#         for w in range(n_w):
#             kernel[h, w] = (1 / (np.sqrt(2 * np.pi) * sigma_h * sigma_w)) * np.exp(-(((h - center_h)**2 / (2 * sigma_h**2)) + ((w - center_w)**2 / (2 * sigma_w**2))))

#     # Normalize the kernel so that the values sum to 1
#     kernel = kernel / np.sum(kernel)

#     return kernel

# def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sig = 1):
#     """
#     It produces single gaussian at expected center
#     :param center:  the mean position (X, Y) - where high value expected
#     :param image_size: The total image size (width, height)
#     :param sig: The sigma value
#     :return:
#     """
#     x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]

#     y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
#     xx, yy = np.meshgrid(x_axis, y_axis)
  
#     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
   
#     return kernel
# a = np.zeros(shape=(383,231,9))
# a[int(50),int(25)] = 0.9
# k1 = gaussian_heatmap(center =(2,2),image_size=(4,4),sig=1)


# k2 = gaussian_2d_kernel(center =(int(383/4),int(231/4)),k_size=(int(383/2),int(231/2)),sig=[1,1])

# k3 = gkern(4,1)
# a1 = a
# for i in range(4):
#     a[:,:,0] = ndimage.convolve(a[:,:,0], k2, mode='constant')
#     print(np.unravel_index(np.argmax(a[:,:,0],axis=None),a[:,:,0].shape))



# # b = ndimage.convolve(a[:,:,0], k, mode='constant')
# # original_resolution = 0.05 # m/pix
# # original_width = 383
# # original_height = 231
# # # to convert from pixel to world is  delta pix * scale (1 pix = 0.05m)
# # # whe need to convert this scale to a new one
# # #map width in meters is 19.15 m -> 19.15 m/192  = 0.09973958333
# # #map height in meters is 11.51 m -> 11.51 m/116 = 0.09922413793
# # #if map shrinks the scale multiplies
# # # scale = int(ker_weidth/original_weidth)
# # # resolution = org res * scale

# # #We first were at the center (0,0)
# # #Then we moved two pixels to the right (-y) (-2,0)
# # #116,192

# # # a = np.random.randint(10, size=(383,231))
# # a = np.zeros(shape=(383,231,9))
# # a[int(383/2),int(231/2)] = 10000000
# # # k = gaussian_heatmap(center =(60,96),image_size=(116,192),sig=1)
# # k = gaussian_2d_kernel(center =(58,90),k_size=(116,192),sig=[1,1])

# # b = ndimage.convolve(a[:,:,0], k, mode='constant')

# # k2 = gauss_1d_kernel(4,1,0)

# # a[:,:,0] = b
# # a[int(383/2),int(231/2),1:] = [0,0,0,0.9,0,0,0,0]

# # for x in range(a.shape[0]):
# #     for y in range(a.shape[1]):        
# #         out = ndimage.convolve1d(a[x,y,1:], k2)
       
# #         a[x,y,1:] = out
# # # out = ndimage.convolve1d([10000000, 10000000, 10000000, 10000000, 80000000, 10000000, 10000000,
# # #  10000000], k2)
# # # print(a[int(383/2),int(231/2),1:])
# # # # out = ndimage.convolve1d(x, k2)

# # # # y si primero hacemos el 1d y luego el 2d ? nah porque el primer numero afecta , asi esta bien

# fig=plt.figure()
# fig.add_subplot(1,3,1)
# plt.imshow(k1, cmap="Reds", interpolation='nearest')
# fig.add_subplot(1,3,2)
# plt.imshow(a1[:,:,0], cmap="Reds", interpolation='nearest')
# fig.add_subplot(1,3,3)
# plt.imshow(a[:,:,0], cmap="Reds", interpolation='nearest')
# plt.show()

# # # grid = np.zeros(100)
# # # n = grid.size
# # # grid[:] = 1. / n
# # # u = np.array([
# # #     [0.1, 0.7, 0.2, 0.0, 0.0],  # Move one to the left under uncertainty
# # #     [0,0, 0.0, 0.2, 0.7, 0.1]   # Move one to the right under uncertainty    
# # # ])
# # # print(grid)

# # # grid[:] = ndimage.convolve(grid, u[1], mode='constant')
# # # print(grid)




# # # # def uniform_prior(grid):
# # # #     """Initializes the grid with an uniform prior."""
# # # #     n = grid.size
# # # #     grid[:] = 1. / n

# # # # def predict(grid, u):
# # # #     """Perform prediction step using movement command `u`.
    
# # # #     Updates the belief of being in a specific state given the movement
# # # #     command and previous state. Based on the law of total probability,
# # # #     we get for the new state

# # # #         P(x_t) =  sum P(x_t|x_t-1, u) * P(x_t-1)
# # # #                  x_t-1

# # # #     Meaning that for each possible state we need to sum over all the
# # # #     probable ways x_t could have been reached from x_t-1 times the
# # # #     prior probability of being in x_t-1.

# # # #     In signal theory this corresponds to performing a convolution
# # # #     of the signal P(x_t-1) and the probabilistic signal u. Note that
# # # #     convolution rotates the signal u by 180°, because we are interested
# # # #     in the ways x_t could have been reached from x_t-1 (inverse motion).
# # # #     """
# # # #     grid[:] = ndimage.convolve(grid, u, mode='constant')
 
# # # def correct(grid, z, stddev=1.):
# # #     """Perform correction/measurement step.
    
# # #     Updates the belief state by incorporating a measurement. The measurement
# # #     updated is given by Bayes rule

# # #         P(x_t|z) = n * P(z|x_t) * P(x_t)

# # #     Here n is a normalizer to make P(x_t|z) a PMF given by

# # #         n = sum P(z|xi_t) * P(xi_t)
# # #              i
# # #     """
# # #     n = 0.
# # #     for i in range(grid.shape[0]):
# # #         # Should get a reading of i (measured towards wall at zero), got z 
# # #         alpha = norm.pdf(z, loc=i, scale=stddev) * grid[i]
# # #         print(alpha)
# # #         grid[i] = alpha
# # #         n += alpha

# # #     grid /= n
# # #     print(grid)
# # # correct(grid=grid,z=60)
# # # # u = np.array([
# # # #     [0.1, 0.7, 0.2, 0.0, 0.0],  # Move one to the left under uncertainty
# # # #     [0,0, 0.0, 0.2, 0.7, 0.1]   # Move one to the right under uncertainty    
# # # # ])

# # # # grid = np.zeros(100)
# # # # # print(grid.shape)
# # # # # grid = grid.reshape(1, -1)
# # # # # print(grid.shape)
# # # # moves = np.random.randint(0, 2, size=100)
# # # # print(moves)

# # # # fig, ax = plt.subplots()
# # # # im = ax.imshow(grid.reshape(1, -1), interpolation='none', cmap='hot', extent=[0, grid.size, 0, 1], vmin=0, vmax=1, aspect='auto', animated=True)
# # # # line, = ax.plot((0, 0), (0, 1), 'r-')


# # # # def init():
# # # #     global pos

# # # #     pos = 50
# # # #     uniform_prior(grid)

# # # #     im.set_array(grid.reshape(1, -1))
# # # #     line.set_xdata((pos,pos))
# # # #     return im, line

# # # # def update(i):
# # # #     global pos

# # # #     m = moves[i % len(moves)]
# # # #     print(i % len(moves))
  
# # # #     pos += -1 if m == 0 else 1
# # # #     predict(grid, u[m])  

# # # #     if i % 20 == 0:
# # # #         correct(grid, pos, stddev=2.)

# # # #     im.set_array(grid.reshape(1, -1))
# # # #     line.set_xdata((pos,pos))
# # # #     return im, line

# # # # ani = FuncAnimation(fig, update, init_func=init, interval=200, frames=len(moves), repeat=False, blit=True)
# # # # plt.show()