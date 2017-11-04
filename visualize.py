import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from frames_dataset import FramesDataset


def show_frame(frame, heat_transfer):
    plt.imshow(frame,cmap='gray')


if __name__ == "__main__":
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    help(face_dataset)
    fig = plt.figure()

    num_samples = 1
    samples_indexes = np.random.randint(len(face_dataset), size=num_samples)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        print(i, sample['frame'].shape, sample['heat_transfer'].shape)

        ax = plt.subplot(num_samples//3+1, 3, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(index))
        ax.axis('off')

        #тензор для суммы всех картинок
        AllPicTensor=torch.LongTensor()

        #накладываем фильтр поиска границ Лапласиан
        LaplacianFilterSample = cv2.Laplacian(sample['frame'], 3)

        #передаем картинку в виде numpyArray в map структуру которую может показывать функция show
        sample['frame']=LaplacianFilterSample

        #показываем картинку на экране из map структуры
        show_frame(**sample)

        #делаем Тензон из NumpyArray что бы не было ошибок тип данных массива должен совпадать с типом данных тензора, сам он не меняет тип
        TensorSample=torch.ShortTensor(LaplacianFilterSample)

        #преврящаем тип данных в тенозоре в Long т.к. для сложения всех матриц Short не хватит, а разные типы складывать нельзя в pytorch
        TensorSample.long()

        #помещаем тензор в память видеокарты
        TensorSample = TensorSample.cuda()

        #проверяем находится ли тензор в памяти видеокарты, (выводит True если тензор в видеопамять)
        print(TensorSample.is_cuda)

        if i == num_samples-1:
            plt.show()
            break
