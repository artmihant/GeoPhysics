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
    name = "delta_model_8_energy"
    filepath_alpha = '/home/artem/Projects/GeoPhysics/postprocessing/data/alpha_s8_{}.sgy'
    filepath_beta = '/home/artem/Projects/GeoPhysics/postprocessing/data/beta_s8_{}.sgy'
    filepath_gamma = '/home/artem/Projects/GeoPhysics/postprocessing/data/full_s8_8_{}.sgy'

    # filepath_alpha_out = '/home/artem/Projects/GeoPhysics/postprocessing/data/compare_alpha_model_8_{}.sgy'
    # filepath_beta_out = '/home/artem/Projects/GeoPhysics/postprocessing/data/compare_beta_model_8_{}.sgy'
    # filepath_gamma_out = '/home/artem/Projects/GeoPhysics/postprocessing/data/cut8_8{}.sgy'

    proc_size = (22, 300, 141, 50)

    # alpha_size = (52, 300, 231, 50)
    alpha_size = (22, 300, 141, 50)

    beta_size = (22, 300, 141, 50)

    frames = 1500

def segy_read(filepath, size):

    width, x_step, height, y_step = size
    proc_width, proc_x_step, proc_height, proc_y_step = CONFIG.proc_size

    data = np.zeros((CONFIG.frames, proc_width, proc_height, 3), dtype=np.float32)

    for axis in range(3):
        with segyio.open(filepath.format(axis), 'r') as segyfile:

            assert segyfile.ilines.size == width * height, f'{segyfile.ilines.size} not equal {width} * {height}!'

            for index in range(len(segyfile.trace)):
                x,y = (index//height), (index%height)

                x = ((2*x-width+1)*x_step/proc_x_step + (proc_width-1))/2
                y = ((2*y-height+1)*y_step/proc_y_step + (proc_height-1))/2

                if int(x) != x or int(y) != y:
                    continue

                if 0 > x or x >= proc_width or 0 > y or y >= proc_height:
                    continue

                data[:, int(x), int(y), axis] = segyfile.trace[index]

    return data

def segy_white(filepath, data):

    proc_width, proc_x_step, proc_height, proc_y_step = CONFIG.proc_size

    for axis in range(3):
        with segyio.open(filepath.format(axis), 'r+') as segyfile:

            for index in range(len(segyfile.trace)):
                x,y = (index//proc_height), (index%proc_height)

                segyfile.trace[index] = data[:, x, y, axis]

def main():


    data_alpha = segy_read(CONFIG.filepath_alpha, CONFIG.alpha_size) 

    print(data_alpha.min(),data_alpha.max())

    data_beta = segy_read(CONFIG.filepath_beta, CONFIG.beta_size) 

    print(data_beta.min(),data_beta.max())

    data_alpha_beta_delta = data_alpha - data_beta
    print(data_alpha_beta_delta.min(), data_alpha_beta_delta.max())


    # segy_white(CONFIG.filepath_gamma, data_alpha) 
    
    # print(data_gamma.min(),data_gamma.max())



    # data_alpha_gamma_delta = data_alpha - data_gamma
    # print(data_alpha_gamma_delta.min(), data_alpha_gamma_delta.max())

    # data_beta_gamma_delta = data_beta - data_gamma
    # print(data_beta_gamma_delta.min(), data_beta_gamma_delta.max())



if __name__ == '__main__':

    main()


