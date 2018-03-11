import matplotlib.pyplot as plt
import torch
import numpy
from torch.autograd import Variable
import numpy as np

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from picture_transformation import init_edge_feature_map_5x5

def capture_feature(face_dataset, feature_map, num_sample,from_layer, layers_by_wall):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param feature_map: array of with feature map 1 dimension-number of temple, 2-3 dimentions size temple of picture
    :param num_samples: number of picture to calculation
    :param layers_by_wall: number of layers by wall where i search features
    :param from_layer: number of layers from edge of picture by Y
    """
    #final results matrix will has lower second two dimension(7x340) than initial pictures (48x340) because we use only part by the wall, first dimension is equal feature dimension of feature_map
    size_pic_X=face_dataset[num_sample]['frame'].shape[1]
    size_pic_Y=face_dataset[num_sample]['frame'].shape[0]
    feature_size_X=feature_map.shape[2]
    feature_size_Y=feature_map.shape[1]
    feature_amount=feature_map.shape[0]
    SummResult = torch.FloatTensor(feature_amount,layers_by_wall-from_layer,size_pic_X).zero_()
    one_dimension_result = torch.FloatTensor(1,1,feature_amount*size_pic_X*(layers_by_wall-from_layer)).zero_()

    if torch.cuda.is_available():
        SummResult.cuda()
        one_dimension_result.cuda()

    heating_map= np.zeros(shape=(size_pic_Y, size_pic_X), dtype='int32')

    #here i extract boundaries from sample, format binares picture
    BinareFilterSample = boundaries_detect_laplacian(face_dataset[num_sample])/255
    if torch.cuda.is_available(): BinareFilterSample.cuda()

    #here i compare k features_pictures with layer on the sample picture with srtide 1 px
    for k in range(0,feature_amount):
        for i in range(0,layers_by_wall-from_layer):
            for j in range(0, size_pic_X-feature_size_X):
                #here i calculate coordinates clice of sample to compare with feature map templets
                x1=j
                x2=j+feature_size_X
                y1=size_pic_Y-feature_size_Y-(i+from_layer)
                y2=size_pic_Y-(i+from_layer)
                local_feature = BinareFilterSample[y1:y2, x1:x2]
                #here i multiply pixel by pixel,this operation save only nonzero elements at both matrix, futher i summ all nonzero elements
                SummResult[k,i,j] = torch.sum(local_feature*feature_map[k])+1
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
        TensorSample =BinareFilterSample

        # We change the data type in the tensor to the Long type because to add all the matrices Short is not enough, and different types can not be added in pytorch
        TensorSample = TensorSample.long()

        # multiply by 1000 to allocate borders in the total total amount
        SummResult.add_(TensorSample)

        break
    return SummResult

if __name__ == "__main__":

    #test run LSTM from manual example
    target=Variable(torch.FloatTensor(5,3,20))
    rnn = torch.nn.LSTM(10, 20, 2)
    rnn.cuda()
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
    #face_dataset = FramesDataset('file:///media/aleksandr/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/aleksandr/Files/@Machine/Github/Boiler/train')
    face_dataset = FramesDataset('./train/annotations.csv', './train')
    print('dataset is loaded')
    #here i init feature map and put it to the videomemory(.cuda)
    feature_map = init_edge_feature_map_5x5()
    if torch.cuda.is_available():
        feature_map.cuda()


    test='true'
    if test=='true':
        #visualize some data
        sample = face_dataset[1]
        fig = plt.figure()
        print(1, sample['frame'].shape, sample['heat_transfer'].shape)
        for j in range(0, 1):
            for i in range(1, 3):
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

    hidden_features=100
    number_of_samples_lstm=120000
    first_sample_lstm=28*12000 #63 * 12000
    error=numpy.zeros(shape=(number_of_samples_lstm), dtype='float32')
    input = Variable(capture_feature(face_dataset, feature_map, 0,0, 10)[1])
    target=torch.FloatTensor(first_sample_lstm+number_of_samples_lstm)
    # number of features input, number of features hidden layer ,2- number or recurent layers
    rnn = torch.nn.LSTM(input.data.shape[2], hidden_features, 1)
    ln=torch.nn.Linear(100,1)
    #rnn.weight_ih_l0.data.fill_(1000)
    #rnn.weight_hh_l0.data.fill_(1000)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    h0 = Variable(torch.FloatTensor(1,1,hidden_features))
    h0.data[0,0,0]=120000
    c0 = Variable(torch.FloatTensor(1,1,hidden_features))
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
    for sample_num in range(first_sample_lstm, first_sample_lstm + number_of_samples_lstm):
        print(sample_num)
        target[sample_num] = float(face_dataset[sample_num]['heat_transfer'])
        #print(sample_num)
    target=Variable(target)
    print('heat transfer hedings are loaded')
    print(target)
    #cycle by all samples
    for sample_num in range(first_sample_lstm, first_sample_lstm+number_of_samples_lstm):
        output, (hn, cn) = rnn(input)
        output=ln(output)
        w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
        w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
        loss = (torch.sum(output)-(target[sample_num]))
        error[sample_num-first_sample_lstm]=loss.data[0]
        input=Variable(capture_feature(face_dataset, feature_map, sample_num,0,10)[1])
        loss.backward()
        optimizer.step()
        ln.weight.data = ln.weight.data + ln.weight.grad.data*0.01#*(0.01*abs(loss.data[0])+0.01)
        print(str(sample_num-first_sample_lstm)+' '+str(torch.sum(w_ii - u_ii).data[0]) + '  ' + str(torch.sum(w_if - u_if).data[0]) + '  ' + str(
            torch.sum(w_ic - u_ic).data[0]) + '  ' + str(torch.sum(w_io - u_io).data[0]) + '  ' + str(
            torch.sum(w_hi - u_hi).data[0]) + '  ' + str(torch.sum(w_hf - u_hf).data[0]) + '  ' + str(
            torch.sum(w_hc - u_hc).data[0]) + '  ' + str(torch.sum(w_ho - u_ho).data[0]) + '  loss=' + str(
            loss.data[0])+'  out'+str(torch.sum(output).data[0])+'   '+str(torch.mean(ln.weight).data[0])+'  '+ str(torch.mean(ln.weight.grad).data[0]))
        ln.weight.grad.data.zero_()
        optimizer.zero_grad()

    print(u_ii)
    print(u_if)
    print(u_ic)
    print(u_io)
    print('max io '+str(torch.max(u_io))+'min io '+str(torch.min(u_io)))
    print(u_hi)
    print(u_hf)
    print(u_hc)
    print(u_ho)

    plt.clf()
    plt.axes([0.3, 0.3, 0.5, 0.5])
    # plt.title ('iteration error for ' + str(5) +' max eigenvalues and eigenvectors')
    plt.plot(error, 'k:', label='1')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.legend()
    plt.show()