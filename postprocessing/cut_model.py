import os
import subprocess
import numpy as np
from PIL import Image as im 
import segyio
import shutil
import time

@np.vectorize
def to_pixel(x, k):
    x = x*255*k
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return int(x)


class CONFIG:
    path = '/home/artem/Projects/MSU270/segy/data/s6_model_{}_{}.sgy'
    projectpath = '/home/artem/Projects/MSU270/segy'

@np.vectorize
def to_pixel(x, k):
    x = x*255*k
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return int(x)

def draw(data, name):

    image = np.zeros((400,1500,3), dtype=np.uint8)
    image[:,:,0] = to_pixel(data,1).reshape(image.shape[:-1])
    image[:,:,1] = to_pixel(data,49).reshape(image.shape[:-1])
    image[:,:,2] = to_pixel(data,7).reshape(image.shape[:-1])
    im.fromarray(image).save(os.path.join(CONFIG.projectpath, name))

def main():
    
    data_alpha = np.zeros((400, 1500, 3),dtype=np.float32)
    data_beta = np.zeros((400, 1500, 3),dtype=np.float32)
    
    for alpha in range(3,10):
        beta = alpha + 1

        for axis in range(3):

            with segyio.open(CONFIG.path.format(alpha, axis), 'r') as alpha_file:
                for index in range(400):
                    data_alpha[index,:,axis] = alpha_file.trace[index]
                    # data_alpha[index,:,axis] = (alpha.trace[400*199+index] + alpha.trace[400*200+index])/2

            with segyio.open(CONFIG.path.format(beta, axis), 'r') as beta_file:
                for index in range(400):
                    data_beta[index,:,axis] = beta_file.trace[index]
                    # beta.trace[index] = data_alpha[index,:,axis]
                    # data_beta[index,:,axis] = (beta.trace[400*199+index] + beta.trace[400*200+index])/2


        energy_alpha = (data_alpha[:,:,0]**2+data_alpha[:,:,1]**2+data_alpha[:,:,2]**2).reshape(-1,1)
        energy_beta = (data_beta[:,:,0]**2+data_beta[:,:,1]**2+data_beta[:,:,2]**2).reshape(-1,1)

        energy_delta = (
            (data_alpha[:,:,0]-data_beta[:,:,0])**2+
            (data_alpha[:,:,1]-data_beta[:,:,1])**2+
            (data_alpha[:,:,2]-data_beta[:,:,2])**2
        ).reshape(-1,1)

        draw(energy_alpha, f'images/d_{alpha}.png')
        draw(energy_beta, f'images/d_{beta}.png')
        draw(energy_delta, f'images/dl_{alpha}_{beta}.png')

if __name__ == '__main__':

    main()
