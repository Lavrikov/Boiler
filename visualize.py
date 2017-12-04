import matplotlib.pyplot as plt
import torch

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from train_k_neihbor import boundaries_summ_conv

def show_frame(frame, heat_transfer):
    plt.imshow(frame,'gray')


if __name__ == "__main__":
    i=1
    #load the video dataset like a group of a pictures
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    #calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
    SummResult=boundaries_summ_conv(face_dataset,13 * 12000, 13 * 12000 + 8, 1000)

    sample=face_dataset[i]
    fig = plt.figure()
    print(i, sample['frame'].shape, sample['heat_transfer'].shape)
    ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #coordinates
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(1))
    ax.axis('off')
    print(SummResult)

    # show the statistic matrix
    sample['frame'] = SummResult.numpy()
    show_frame(**sample)
    plt.show()

