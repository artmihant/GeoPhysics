import numpy as np
from PIL import Image as im 

center = [
    439021.250000,
    7839164.000000, 
    0
]

def to_pixel(x):
    x = x*255
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return int(x)

def lut_function(data):

    r = 0
    g = 0
    b = 0

    energy = data[0]**2 + data[1]**2 + data[2]**2

    r = to_pixel(data[2] if data[2] > 0 else 0)

    g = to_pixel((data[0]**2 + data[1]**2 + data[2]**2)*50)

    b = to_pixel(-data[2] if data[2] < 0 else 0)

    return np.array([r,g,b], dtype=np.uint8)


def prepare(filepath):

    data = np.zeros((1500, 280, 280, 3), dtype=np.float64)

    for j in range(3):
        with open(filepath.format(j), 'r+') as rf:

            line = next(rf).split()

            for i, src_line in enumerate(rf):
                line = np.array(src_line.split()[1:], dtype=np.float64).reshape(280, 280)
                data[i,:,:,j] = line
                i+=1

    print(data[477])

    # for i in range(1500):
    #     print(i, data[i].min(), data[i].max())

    # image = np.zeros((280, 280, 3), dtype=np.uint8)

    # for t in range(0, 1500):

    #     for i in range(data.shape[1]):
    #         for j in range(data.shape[2]):
    #             # if i == 0 or i == data.shape[1]-1 or j == 0 or j == data.shape[2]-1:
    #             #     pvalue = np.array([255,255,255], dtype=np.uint8)
    #             # else:
    #             pvalue = lut_function(data[t,i,j])
    #             image[i,j] = pvalue


    #     im.fromarray(image[::-1]).save(f'images2/frame_{t:04d}.png') 

    #     print(t, image.max())

    # assert np.isfinite(energy).all() 

    pass



if __name__ == '__main__':
    filepath = '/home/artem/Projects/MSU270/segy/data/c8_model_4_{}.txt'

    prepare(filepath)


# ffmpeg -framerate 25 -i images/frame_%04d.png output.mp4
# 
# ffmpeg -r 1/5 -i img%03d.png -c:v libx264 -vf fps=50 -pix_fmt yuv420p out.mp4
