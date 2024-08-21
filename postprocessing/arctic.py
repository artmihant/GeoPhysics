import os
import subprocess
import numpy as np
from PIL import Image as im 
import segyio
import shutil
import time
import segyio.su


class CONFIG:
    path = '/home/artem/Projects/MSU270/SEM3D/data/src/Rho_model3D_60km2.sgy'
    projectpath = '/home/artem/Projects/MSU270/segy'


def main():
    

    with segyio.open(CONFIG.path, 'r') as alpha_file:

        # print(alpha_file.trace[0])
        # print(alpha_file.trace[1])
        # print(alpha_file.trace[2])
        for i in range(100):
            print(alpha_file.header[i], alpha_file.header[i][segyio.su.sx], alpha_file.header[i][segyio.su.sy])
        # print(alpha_file.header[1])
        # print(alpha_file.header[2])
        # print(alpha_file.header[3])
        # print(alpha_file.header[1])

        # print(alpha_file.header)

        pass

        alpha_file

        # for index in range(400):
        #     data_alpha[index,:,axis] = alpha_file.trace[index]
        #     # data_alpha[index,:,axis] = (alpha.trace[400*199+index] + alpha.trace[400*200+index])/2

        # energy_alpha = (data_alpha[:,:,0]**2+data_alpha[:,:,1]**2+data_alpha[:,:,2]**2).reshape(-1,1)
        # energy_beta = (data_beta[:,:,0]**2+data_beta[:,:,1]**2+data_beta[:,:,2]**2).reshape(-1,1)

        # energy_delta = (
        #     (data_alpha[:,:,0]-data_beta[:,:,0])**2+
        #     (data_alpha[:,:,1]-data_beta[:,:,1])**2+
        #     (data_alpha[:,:,2]-data_beta[:,:,2])**2
        # ).reshape(-1,1)

        # draw(energy_alpha, f'images/d_{alpha}.png')
        # draw(energy_beta, f'images/d_{beta}.png')
        # draw(energy_delta, f'images/dl_{alpha}_{beta}.png')

if __name__ == '__main__':

    main()
