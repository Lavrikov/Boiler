import matplotlib.pyplot as plt
import torch

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from train_k_neihbor import boundaries_summ_conv

def show_frame(frame, heat_transfer):
    plt.imshow(frame,'gray')


if __name__ == "__main__":
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    SummResult=boundaries_summ_conv(face_dataset)
    sample=face_dataset[1]
    fig = plt.figure()
    print(i, sample['frame'].shape, sample['heat_transfer'].shape)
    ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #здесь устанавливаются координаты
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(1))
    ax.axis('off')

    #Делим на число кадров
    #SummResult=SummResult/num_samples

     print(SummResult)

     # показываем картинку на экране из map структуры
     sample['frame'] = SummResult.numpy()
     show_frame(**sample)
     plt.show()

