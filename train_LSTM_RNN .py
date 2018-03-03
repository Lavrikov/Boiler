import matplotlib.pyplot as plt
import torch
import numpy
from torch.autograd import Variable
import numpy as np

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from picture_transformation import init_edge_feature_map_5x5

def capture_feature(face_dataset, feature_map, num_sample, layers_by_wall):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param feature_map: array of with feature map 1 dimension-number of temple, 2-3 dimentions size temple of picture
    :param num_samples: number of picture to calculation
    :param number of layers by wall where i search features
    """
    #final results matrix will has lower second two dimension(7x340) than initial pictures (48x340) because we use only part by the wall, first dimension is equal feature dimension of feature_map
    size_pic_X=face_dataset[num_sample]['frame'].shape[1]
    size_pic_Y=face_dataset[num_sample]['frame'].shape[0]
    feature_size_X=feature_map.shape[2]
    feature_size_Y=feature_map.shape[1]
    feature_amount=feature_map.shape[0]
    SummResult = torch.FloatTensor(feature_amount,layers_by_wall,size_pic_X).zero_()
    one_dimension_result = torch.FloatTensor(1,1,feature_amount*size_pic_X*layers_by_wall).zero_()
    heating_map= np.zeros(shape=(size_pic_Y, size_pic_X), dtype='int32')

    #here i extract boundaries from sample, format binares picture
    BinareFilterSample = boundaries_detect_laplacian(face_dataset[num_sample])/255

    #here i compare k features_pictures with layer on the sample picture with srtide 1 px
    for k in range(0,feature_amount):
        for i in range(0,layers_by_wall):
            for j in range(0, size_pic_X-feature_size_X):
                #here i calculate coordinates clice of sample to compare with feature map templets
                x1=j
                x2=j+feature_size_X
                y1=size_pic_Y-feature_size_Y-i
                y2=size_pic_Y-i
                local_feature = BinareFilterSample[y1:y2, x1:x2]
                #here i multiply pixel by pixel,this operation save only nonzero elements at both matrix, futher i summ all nonzero elements
                SummResult[k,i,j] = np.sum(local_feature*feature_map[k])+1
                one_dimension_result[0,0,k*i+j]=SummResult[k,i,j]+1
                if SummResult[k,i,j]>4: heating_map[y1,x1]=heating_map[y1,x1]+SummResult[k,i,j]
    test='false'
    if test=='true':
        # here i show results of feature_map impacting
        fig = plt.figure()
        ax = plt.subplot(2, 1, 1)  # coordinates
        plt.tight_layout()
        ax.set_title('Sample #{}'+str(num_sample))
        ax.axis('off')
        plt.imshow(BinareFilterSample, 'gray')
        ax = plt.subplot(2, 1, 2)  # coordinates
        plt.tight_layout()
        ax.set_title('Sample #{}'+str(num_sample))
        ax.axis('off')
        # print(SummResult)
        # show the statistic matrix
        plt.imshow(heating_map, 'gray')
        plt.show()

    return SummResult, one_dimension_result

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

        # do Tenson from NumpyArray that there were no errors the data type of the array must match the data type of the tensor, it does not change the type by himself
        TensorSample = torch.from_numpy(BinareFilterSample)

        # We change the data type in the tensor to the Long type because to add all the matrices Short is not enough, and different types can not be added in pytorch
        TensorSample = TensorSample.long()

        # multiply by 1000 to allocate borders in the total total amount
        SummResult.add_(TensorSample)
        print(SummResult)
        break
    return SummResult

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

    #test run LSTM from manual example
    target=Variable(torch.FloatTensor(5,3,20))
    rnn = torch.nn.LSTM(10, 20, 2)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    input = Variable(torch.randn(5, 3, 10))
    h0 = Variable(torch.randn(2, 3, 20))
    c0 = Variable(torch.randn(2, 3, 20))
    output, hn = rnn(input, (h0, c0))
    u_ii=1
    for j in range(0, 10):
        output, hn = rnn(input)
        w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
        loss=torch.mean(output-target)
        loss.backward()
        optimizer.step()
        print(torch.sum(w_ii-u_ii))
        u_ii = w_ii.clone()
        #print(w_ii)
        #print(w_ii.grad)
        #print(loss)

    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset('file:///media/aleksandr/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/aleksandr/Files/@Machine/Github/Boiler/train')

    #here i init feature map
    feature_map = init_edge_feature_map_5x5()

    test='false'
    if test=='true':
        #visualize some data
        sample = face_dataset[1]
        fig = plt.figure()
        print(1, sample['frame'].shape, sample['heat_transfer'].shape)
        for j in range(0, 1):
            for i in range(1, 13):
                # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
                SummResult = boundaries_summ_conv(face_dataset, 63 * 12000 + i + j * 13, 63 * 12000 + i + 40 + j * 13, 1000)
                # here i show results
                ax = plt.subplot(11 // 3 + 1, 3, i)  # coordinates
                plt.tight_layout()
                ax.set_title('Sample #{}'.format(i))
                ax.axis('off')
                # show the statistic matrix
                plt.imshow(SummResult.numpy(), 'gray')
                SummResult.zero_()

            plt.show()

    # here i init NN
    # The structure of input tensor LSTM fo picture recognition (seq-lenght, batch, input_size)
    # 1 argument - is equal to number of pictures- for every picture is calculated h-t and putted to output
    # 2 argument - is equal a number of batch - better to use 1
    # 3 argument - is one dimensional array contains all features from all part of picture- it is require convert 3 dimensional array to 1 dimension and put to this cell.
    # The structure of paramenters LSTM(lenght of array with features=big value, number of heatures in output can be lower and higer than in input- how mach i want, number of layer in recurent model)
    input = Variable(capture_feature(face_dataset, feature_map, 0, 10)[1])
    target=Variable(torch.FloatTensor(1,1,1))
    target.data[0,0,0]=float(face_dataset[63 * 12000]['heat_transfer'])
    # number of features input, number of features hidden layer ,2- number or recurent layers
    rnn = torch.nn.LSTM(input.data.shape[2], 1, 1)
    #rnn.weight_ih_l0.data.fill_(1000)
    #rnn.weight_hh_l0.data.fill_(1000)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    h0 = Variable(torch.FloatTensor(1,1,1))
    h0.data[0,0,0]=120000
    c0 = Variable(torch.FloatTensor(1,1,1))
    c0.data[0, 0, 0] = 120000
    output, (hn, cn) = rnn(input, (h0, c0))
    w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
    u_ii = w_ii.clone()
    u_if = w_if.clone()
    u_ic = w_ic.clone()
    u_io = w_io.clone()
    u_hi = w_hi.clone()
    u_hf = w_hf.clone()
    u_hc = w_hc.clone()
    u_ho = w_ho.clone()
    #cycle by all samples                                                                                                                                                                                   
    for j in range(0, 1000):
        sample_num=63 * 12000+j

        output, (hn, cn) = rnn(input)
        w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
        #target = Variable(float(face_dataset[sample_num]['heat_transfer']))
        loss = (output-target)
        print(str(torch.sum(w_ii-u_ii).data[0])+'  '+str(torch.sum(w_if-u_if).data[0])+'  '+str(torch.sum(w_ic-u_ic).data[0])+'  '+str(torch.sum(w_io-u_io).data[0])+ '  '+ str(torch.sum(w_hi-u_hi).data[0])+'  '+str(torch.sum(w_hf-u_hf).data[0])+'  '+str(torch.sum(w_hc-u_hc).data[0])+'  '+str(torch.sum(w_ho-u_ho).data[0])+'  loss='+str(loss.data[0,0,0]) )
        if (torch.sum(w_io-u_io).data[0])>0:
            input=Variable(capture_feature(face_dataset, feature_map, sample_num, 10)[1])
        #print(loss)
        #print(str(j)+'      '+ str(loss.data[0,0,0])+'      '+ str(torch.mean(w_ii.data))+'  '+ str(torch.mean(w_if.data))+'  '+ str(torch.mean(w_ic.data))+'  '+ str(torch.mean(w_io.data))+'       '+ str(torch.mean(w_hi.data))+'  '+ str(torch.mean(w_hf.data))+'  '+ str(torch.mean(w_hc.data))+'  '+ str(torch.mean(w_ho.data)))
        loss.backward()
        optimizer.step()

    print(u_ii)
    print(u_if)
    print(u_ic)
    print(u_io)
    print('max io '+str(torch.max(u_io))+'min io '+str(torch.min(u_io)))
    print(u_hi)
    print(u_hf)
    print(u_hc)
    print(u_ho)




    # here i show results
    #sample=output
    #fig = plt.figure()
    #print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    #ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #coordinates
    #plt.tight_layout()
    #ax.set_title('Sample #{}'.format(1))
    #ax.axis('off')
    #print(output)

    # show the statistic matrix
    #plt.imshow(face_dataset[12100]['frame'],'gray')
    #plt.show()