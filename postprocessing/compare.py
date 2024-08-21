import os
import subprocess
import sys
import numpy as np
from PIL import Image as im 
import segyio
import shutil
import time


def main():

    filepath_alpha = '/home/artem/Projects/GeoPhysics/postprocessing/data/alpha_s8_2.sgy'
    filepath_beta = '/home/artem/Projects/GeoPhysics/postprocessing/data/beta_s8_2.sgy'


    samples = 1500
    traces = 22*141

    data_alpha = np.zeros((traces, samples), dtype=np.float32)
    data_beta = np.zeros((traces, samples), dtype=np.float32)

    with segyio.open(filepath_alpha, 'r') as segyfile:

        assert segyfile.ilines.size == traces
        assert segyfile.samples.size == samples

        for index, trace in enumerate(segyfile.trace):
            data_alpha[index] = trace

    with segyio.open(filepath_beta, 'r') as segyfile:
        assert segyfile.ilines.size == traces
        assert segyfile.samples.size == samples

        for index, trace in enumerate(segyfile.trace):
            data_beta[index] = trace

    data_delta = data_beta-data_alpha

    print(data_delta.min(), data_delta.max())
    print(data_alpha[0][1400])
    print(data_beta[0][1400])
    print(data_delta[0][1400])




if __name__ == '__main__':
    main()