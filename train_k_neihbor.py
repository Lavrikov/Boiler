import matplotlib.pyplot as plt
import torch
import numpy as np

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from picture_transformation import init_edge_feature_map_5x5


def boundaries_summ_conv(face_dataset, num_samples_from, num_samples_to, multiply):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param num_samples_from: number of picture to start calculation
    :param num_samples_to: number of picture to end calculation
    :param multiply: coefficient of multiplication boundaries bright
    """

    sample = face_dataset[1]
    SummResult = torch.from_numpy(sample['frame'])
    SummResult = SummResult.long()
    SummResult.zero_()  # pull it by zero

    samples_indexes = [i for i in range(num_samples_from, num_samples_to)]  # A list contains all requires numbers
    print(samples_indexes)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        BinareFilterSample = boundaries_detect_laplacian(sample)

        print(BinareFilterSample)

        # do Tenson from NumpyArray that there were no errors the data type of the array must match the data type of the tensor, it does not change the type by himself
        TensorSample = torch.from_numpy(BinareFilterSample)
        print(TensorSample)

        # We change the data type in the tensor to the Long type because to add all the matrices Short is not enough, and different types can not be added in pytorch
        TensorSample = TensorSample.long()
        print(TensorSample)

        # multiply by 1000 to allocate borders in the total total amount
        SummResult.add_(TensorSample)
        print(SummResult)
        break
    return SummResult

if __name__ == "__main__":

    rt=init_edge_feature_map_5x5()
    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset('file:///media/aleksandr/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/aleksandr/Files/@Machine/Github/Boiler/train')

    sample = face_dataset[1]
    fig = plt.figure()
    print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    for j in range(0, 3):
        for i in range(1, 13):
            # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
            SummResult=boundaries_summ_conv(face_dataset,63 * 12000+i+j*13, 63 * 12000+i+40+j*13, 1000)
            # here i show results
            ax = plt.subplot(11 // 3 + 1, 3, i) #coordinates
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            #print(SummResult)
            # show the statistic matrix
            plt.imshow(SummResult.numpy(),'gray')
            SummResult.zero_()

        plt.show()