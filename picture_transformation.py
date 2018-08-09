import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def resizeX(input):
    """
    :param input: Nupmy array
    """
    size=( int(input.shape[1]/4),input.shape[0])
    output=cv2.resize(input,size)
    return output

def boundaries_detect_laplacian(sample):

    # select the boundaries of objects in the frame
    LaplacianFilterSample = cv2.Laplacian(sample['frame'], 3)

    # zero all pixels that are not boundaries of objects
    ret, BinareFilterSample = cv2.threshold(LaplacianFilterSample, 25, 255, cv2.THRESH_BINARY)
    BinareFilterSample=torch.from_numpy(BinareFilterSample)

    return BinareFilterSample.byte()


def init_edge_feature_map_5x5():
    #here i make features. features have shape of edges with horizontal line
    size_of_feature=7
    number_of_features=76
    feature_map = np.zeros(shape=(number_of_features, size_of_feature, size_of_feature), dtype='uint8')
    #edges for root have started to rising buble
    number_of_edges=20
    for k in range(0, 17):
        #horizontal line
        feature_map[k,-1, 4:7]=1
        #edge line, k+3 because to mach horizontal  line
        for r in range(0, 10):
            i=(6-int(r*np.sin(((k+3)/number_of_edges)*0.5*3.14)))
            j=(3-int(r*np.cos(((k+3)/number_of_edges)*0.5*3.14)))
            if 7>i>=0 and 7>j>=0: feature_map[k, i, j] = 1

    # edges for root ready to departing buble
    number_of_edges = 20
    for k in range(17, 27):
        # horizontal line
        feature_map[k, -1, 2:7] = 1
        # edge line, k+3 becouse to mach horizontal  line
        for r in range(0, 10):
            i = (6 - int(r * np.sin(((k + 5) / number_of_edges) * 0.5 * 3.14)))
            j = (2 - int(r * np.cos(((k + 5) / number_of_edges) * 0.5 * 3.14)))
            if 7 > i >= 0 and 7 > j >= 0: feature_map[k, i, j] = 1

    #edges for root ready to departing buble
    number_of_edges=20
    for k in range(27, 32):
        #horizontal line
        feature_map[k,-1, 1:7]=1
        #edge line, k+3 becouse to mach horizontal  line
        for r in range(0, 10):
            i=(6-int(r*np.sin(((k+5)/number_of_edges)*0.5*3.14)))
            j=(1-int(r*np.cos(((k+5)/number_of_edges)*0.5*3.14)))
            if 7>i>=0 and 7>j>=0: feature_map[k, i, j] = 1

    #mirror feature map
    for k in range(32, 64):
        feature_map[k]=np.flip(feature_map[k-32],1)

    #two line
    for k in range(64, 68):
        feature_map[k, -1, :] = 1
        feature_map[k, 68-k, :] = 1
    #two inclined line
    for k in range(68, 72):
        feature_map[k, -1, :] = 1
        feature_map[k, 72 - k, 2:5] = 1
        feature_map[k, 72 - k - 1, 0:2] = 1
        feature_map[k, 72 - k + 1, 5:7] = 1
    #mirror feature map
    for k in range(72, 76):
        feature_map[k]=np.flip(feature_map[k-4],1)

    test='false'
    if test=='true':
        # here i show feature_map
        for i in range(0, 36):
            # here i show results
            ax = plt.subplot(6, 6, i+1)  # coordinates
            plt.tight_layout()
            ax.set_title(i)
            ax.axis('off')
            # print(SummResult)
            # show the statistic matrix
            plt.imshow(feature_map[i], 'gray')

        plt.show()
        plt.clf()
        for i in range(36, 72):
            # here i show results
            ax = plt.subplot(6, 6, i+1-36)  # coordinates
            plt.tight_layout()
            ax.set_title(i)
            ax.axis('off')
            # print(SummResult)
            # show the statistic matrix
            plt.imshow(feature_map[i], 'gray')

        plt.show()

        plt.clf()
        for i in range(72, 76):
            # here i show results
            ax = plt.subplot(6, 6, i+1-72)  # coordinates
            plt.tight_layout()
            ax.set_title(i)
            ax.axis('off')
            # print(SummResult)
            # show the statistic matrix
            plt.imshow(feature_map[i], 'gray')

        plt.show()

    return torch.from_numpy(feature_map)