import matplotlib.pyplot as plt
import numpy as np
import math
import os

from frames_dataset import FramesDataset
from random import shuffle


if __name__ == "__main__":


    basePath=os.path.dirname(os.path.abspath(__file__))

    video_length = 12000
    number_of_samples_lstm = 50 * video_length
    first_sample_lstm = 26 * video_length

    number_of_samples_lstm_validation = 19 * video_length
    first_sample_lstm_validation =77 * video_length

    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset(basePath+'/train/annotations.csv',basePath+ '/train')

    x_dim=face_dataset[0]['frame'].shape[1]
    y_dim=face_dataset[0]['frame'].shape[0]
    print(x_dim)
    print(y_dim)

    T= np.float(shape=(x_dim))

