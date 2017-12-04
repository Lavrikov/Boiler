import matplotlib.pyplot as plt
import torch

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian


def boundaries_summ_conv(face_dataset):
    SummResult = torch.LongTensor(48, 340)  # create a tenor for storing the sum of pictures
    SummResult.zero_()  # pull it by zero

    num_samples_from = 13 * 12000
    num_samples = 13 * 12000 + 5
    samples_indexes = [i for i in range(num_samples_from, num_samples)]  # список с последовательной нумерацией
    print(samples_indexes)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        BinareFilterSample = boundaries_detect_laplacian(sample)

        # do Tenson from NumpyArray that there were no errors the data type of the array must match the data type of the tensor, it does not change the type by himself
        TensorSample = torch.from_numpy(BinareFilterSample)

        # We change the data type in the tensor to the Long type because to add all the matrices Short is not enough, and different types can not be added in pytorch
        TensorSample = TensorSample.long()
        print(index)

        # multiply by 1000 to allocate borders in the total total amount
        SummResult.add_(TensorSample * 1000)
        break
    return SummResult
