import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from frames_dataset import FramesDataset


def show_frame(frame, heat_transfer):
    plt.imshow(frame,'gray')


if __name__ == "__main__":
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    fig = plt.figure()

    SummResult = torch.LongTensor(48, 340) # создаем тенозор для хранения суммы картинок
    SummResult.zero_()#заполняем его нулями

    num_samples_from=13*12000
    num_samples = 15*12000
    samples_indexes=[i for i in range(num_samples_from,num_samples)] #список с последовательной нумерацией
    print(samples_indexes)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        #накладываем фильтр поиска границ Лапласиан
        LaplacianFilterSample = cv2.Laplacian(sample['frame'], 3)

        #бинарный фильтр
        ret,BinareFilterSample=cv2.threshold(LaplacianFilterSample,127,255,cv2.THRESH_BINARY)

        #делаем Тензон из NumpyArray что бы не было ошибок тип данных массива должен совпадать с типом данных тензора, сам он не меняет тип
        TensorSample=torch.from_numpy(BinareFilterSample)

        #преврящаем тип данных в тенозоре в Long т.к. для сложения всех матриц Short не хватит, а разные типы складывать нельзя в pytorch
        TensorSample=TensorSample.long()
        print(index)

        #складываем тензоры, умножаем на 1000 для выделения границ в общей итоговой сумме
        SummResult.add_(TensorSample*1000)

        if index == num_samples-1:
            print(i, sample['frame'].shape, sample['heat_transfer'].shape)

            ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #здесь устанавливаются координаты
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(index))
            ax.axis('off')

            #Делим на число кадров
            #SummResult=SummResult/num_samples

            print(SummResult)

            # показываем картинку на экране из map структуры
            sample['frame'] = SummResult.numpy()
            show_frame(**sample)
            plt.show()
            break
