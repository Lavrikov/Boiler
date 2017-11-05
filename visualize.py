import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from frames_dataset import FramesDataset


def show_frame(frame, heat_transfer):
    plt.imshow(frame,cmap='gray')


if __name__ == "__main__":
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    fig = plt.figure()

    SummResult = torch.LongTensor(48, 340) # создаем тенозор для хранения суммы картинок
    SummResult.zero_()#заполняем его нулями
    # помещаем тензор в память видеокарты
    ##SummResult = SummResult.cuda()
    # проверяем находится ли тензор в памяти видеокарты, (выводит True если тензор в видеопамять)
    ##print(str(SummResult.is_cuda)+"SummResult is Cuda?")

    num_samples_from=27*12000
    num_samples = 31*12000
    #samples_indexes = np.random.randint(len(face_dataset), size=num_samples)
    samples_indexes=[i for i in range(num_samples_from,num_samples)] #список с последовательной нумерацией
    print(samples_indexes)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        #накладываем фильтр поиска границ Лапласиан
        LaplacianFilterSample = cv2.Laplacian(sample['frame'], 3)
        #LaplacianFilterSample=sample['frame']

        #передаем картинку в виде numpyArray в map структуру которую может показывать функция show
        #sample['frame']=LaplacianFilterSample

        #делаем Тензон из NumpyArray что бы не было ошибок тип данных массива должен совпадать с типом данных тензора, сам он не меняет тип
        TensorSample=torch.ShortTensor(LaplacianFilterSample)
        #TensorSample=torch.ByteTensor(LaplacianFilterSample)

        #преврящаем тип данных в тенозоре в Long т.к. для сложения всех матриц Short не хватит, а разные типы складывать нельзя в pytorch
        TensorSample=TensorSample.long()
        #помещаем тензор в память видеокарты
        #TensorSample = TensorSample.cuda()
        #проверяем находится ли тензор в памяти видеокарты, (выводит True если тензор в видеопамять)
        #print(TensorSample.is_cuda)
        print(index)

        #складываем тензоры
        SummResult.add_(TensorSample)

        if index == num_samples-1:
            print(i, sample['frame'].shape, sample['heat_transfer'].shape)

            ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #здесь устанавливаются координаты
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(index))
            ax.axis('off')

            #Делим на число кадров
            SummResult=SummResult/num_samples

            # переносим суммарный тензор в оперативную память иначе не работает перевод в numpy array
           # SummResult = SummResult.cpu()
            print(SummResult)

            # показываем картинку на экране из map структуры
            sample['frame'] = SummResult.numpy()
            show_frame(**sample)
            plt.show()
            break
