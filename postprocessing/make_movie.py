import os
import subprocess
import sys
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
    name = "alpha_beta_gap"

    # filepath_alpha = '/home/artem/Projects/GeoPhysics/postprocessing/data/2_227_8_{}.sgy'
    # filepath_beta = '/home/artem/Projects/GeoPhysics/postprocessing/data/2_226_model_8_{}.sgy'

    filepath_alpha = '/home/artem/Projects/GeoPhysics/postprocessing/data/alpha_s8_{}.sgy'
    filepath_beta = '/home/artem/Projects/GeoPhysics/postprocessing/data/beta_s8_{}.sgy'

    projectpath = '/home/artem/Projects/GeoPhysics/postprocessing/movie/'

    # alpha_size = (52, 231)
    # beta_size = (52, 231)

    alpha_size = (22, 141)
    beta_size = (22, 141)

    # pixel = (1, 1)
    pixel = (6, 1)

    # alpha_size = (231, 52)
    # beta_size = (231, 52)

    # pixel = (1, 6)

    duration = 60
    framerate = 25

def measureL0(filed):
    return np.sqrt((filed[:,:,:, 0]**2 + filed[:,:,:, 1]**2 + filed[:,:,:, 2]**2).max())

def measureL1(filed):
    return np.sum(np.sqrt(filed[:,:,:, 0]**2 + filed[:,:,:, 1]**2 + filed[:,:,:, 2]**2))

def measureL2(filed):
    return np.sqrt(np.sum(filed[:,:,:, 0]**2 + filed[:,:,:, 1]**2 + filed[:,:,:, 2]**2))

def make_frames():

    alpha_width, alpha_height = CONFIG.alpha_size
    beta_width, beta_height = CONFIG.beta_size

    px, py = CONFIG.pixel
    frames = CONFIG.framerate*CONFIG.duration

    data_alpha = np.zeros((frames, alpha_width*px, alpha_height*py, 3), dtype=np.float32)
    data_beta = np.zeros((frames, beta_width*px, beta_height*py, 3), dtype=np.float32)

    for axis in range(0,3):
        with segyio.open(CONFIG.filepath_alpha.format(axis), 'r') as segyfile:
            assert segyfile.ilines.size == alpha_width * alpha_height, f'{segyfile.ilines.size} not equal {alpha_width} * {alpha_height}!'
            assert segyfile.samples.size%frames == 0, f'{segyfile.samples.size} not divide {CONFIG.framerate} * {CONFIG.duration}!'

            frequency = segyfile.samples.size//frames

            for index, trace in enumerate(segyfile.trace):
                x,y = px*(index//alpha_height), py*(index%alpha_height)
                data_alpha[:, x:(x+px), y:(y+py), axis] = trace[::frequency].reshape(-1,1,1)

    for axis in range(0,3):
        with segyio.open(CONFIG.filepath_beta.format(axis), 'r') as segyfile:
            assert segyfile.ilines.size == beta_width * beta_height, f'{segyfile.ilines.size} not equal {beta_width} * {beta_height}!'
            assert segyfile.samples.size%frames == 0, f'{segyfile.samples.size} not divide {CONFIG.framerate} * {CONFIG.duration}!'

            frequency = segyfile.samples.size//frames

            for index, trace in enumerate(segyfile.trace):
                x,y = px*(index//beta_height), py*(index%beta_height)
                data_beta[:, x:(x+px), y:(y+py), axis] = trace[::frequency].reshape(-1,1,1)

    data_delta = data_alpha - data_beta

    print(data_alpha.min(), data_beta.min())
    print(data_alpha.max(), data_beta.max())

    # print(measureL0(data_delta))
    # print(measureL0(data_alpha))
    # print(measureL0(data_beta))

    print(measureL0(data_delta)/measureL0(data_alpha))


    # print(measureL1(data_delta))
    # print(measureL1(data_alpha))
    # print(measureL1(data_beta))

    print(measureL1(data_delta)/measureL1(data_alpha))


    # print(measureL2(data_delta))
    # print(measureL2(data_alpha))
    # print(measureL2(data_beta))

    print(measureL2(data_delta)/measureL2(data_alpha))


    image = np.zeros(data_delta.shape[1:], dtype=np.uint8)

    print('Считывание данных завершено')

    for t in range(data_delta.shape[0]):

        energy = (data_delta[t,:,:,0]**2+data_delta[t,:,:,1]**2+data_delta[t,:,:,2]**2).reshape(-1,1)
        image[:,:,0] = to_pixel(energy,1000).reshape(image.shape[:-1])
        image[:,:,1] = to_pixel(energy,49000).reshape(image.shape[:-1])
        image[:,:,2] = to_pixel(energy,7000).reshape(image.shape[:-1])

        im.fromarray(image[::-1]).save(os.path.join(CONFIG.projectpath, CONFIG.name, f'frame_{t:04d}.png'))

    print('Генерация кадров завершена')



if __name__ == '__main__':

    framespath = os.path.join(CONFIG.projectpath, CONFIG.name)

    if os.path.exists(framespath):
        shutil.rmtree(framespath) 
    os.makedirs(framespath)

    make_frames()

    command = ["ffmpeg","-y",
        "-framerate", str(CONFIG.framerate),
        "-i", os.path.join(framespath, "frame_%04d.png"),
        os.path.join(CONFIG.projectpath, f"{CONFIG.name}.mp4")
    ]

    # print(' '.join(command))

    subprocess.run(command)



# ffmpeg -framerate 25 -i energy/frame_%04d.png output2.mp4
# 
# ffmpeg -r 1/5 -i img%04d.png -c:v libx264 -vf fps=50 -pix_fmt yuv420p out.mp4
