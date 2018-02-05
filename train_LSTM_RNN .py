import matplotlib.pyplot as plt
import torch
import numpy
from torch.autograd import Variable

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian


def forward(face_dataset, num_samples_from, num_samples_to, multiply):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param num_samples_from: number of picture to start calculation
    :param num_samples_to: number of picture to end calculation
    :param multiply: coefficient of multiplication boundaries bright
    """
    SummResult=Variable(torch.ByteTensor(num_samples_to, 340, 48), requires_grad=True)
    samples_indexes = [i for i in range(num_samples_from, num_samples_to)]  # A list contains all requires numbers
    print(samples_indexes)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]
        numpy.copyto(face_dataset[index]['frame'], numpy.uint8(boundaries_detect_laplacian(sample)))
        SummResult[i]=Variable(torch.from_numpy(numpy.uint8(boundaries_detect_laplacian(sample))))
    return face_dataset


if __name__ == "__main__":
    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    #here i init NN
    #10 number of features input, 20 number of features hidden layer ,2- number or recurent layers
    fer=torch.from_numpy(face_dataset[1]['frame'])
    print(type(fer))
    rnn = torch.nn.LSTM(10, 20, 2)

    input = Variable(torch.randn(5, 3, 10))
    h0 = Variable(torch.randn(2, 3, 20))
    c0 = Variable(torch.randn(2, 3, 20))
    output, hn = rnn(input, (h0, c0))

    forward(face_dataset, 12000, 12200, 1)
    # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000


    # here i show results
    sample=face_dataset[1]
    fig = plt.figure()
    print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #coordinates
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(1))
    ax.axis('off')
    print(output)

    # show the statistic matrix
    plt.imshow(face_dataset[12100]['frame'],'gray')
    plt.show()