import matplotlib.pyplot as plt
import torch
import numpy
from torch.autograd import Variable
import numpy as np
import math

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from picture_transformation import init_edge_feature_map_5x5
from random import shuffle

def capture_feature(face_dataset, feature_map, num_sample_from, num_sample_to, from_layer, layers_by_wall):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param feature_map: array of with feature map 1 dimension-number of temple, 2-3 dimentions size temple of picture
    :param num_samples_from: number of picture to calculation sequence of pictures
    :param num_samples_to: number of picture to calculation sequence of pictures
    :param layers_by_wall: number of layers by wall where i search features
    :param from_layer: number of layers from edge of picture by Y
    """
    #final results matrix will has lower second two dimension(7x340) than initial pictures (48x340) because we use only part by the wall, first dimension is equal feature dimension of feature_map
    m_pool_1d = torch.nn.MaxPool1d(20)
    size_pic_X = face_dataset[num_sample_from]['frame'].shape[1]
    size_pic_Y = face_dataset[num_sample_from]['frame'].shape[0]
    feature_size_X = feature_map.shape[2]
    feature_size_Y = feature_map.shape[1]
    feature_amount = feature_map.shape[0]
    SummResult = torch.FloatTensor(feature_amount, layers_by_wall - from_layer, size_pic_X).zero_()
    one_dimension_result = torch.FloatTensor(num_sample_to-num_sample_from, 1, feature_amount * size_pic_X * (layers_by_wall - from_layer)).zero_()
    heating_map_numpy = np.zeros(shape=(size_pic_Y, size_pic_X), dtype='int64')
    heating_map = torch.from_numpy(heating_map_numpy)

    for num_sample_in in range(num_sample_from,num_sample_to):


        if torch.cuda.is_available():
            SummResult.cuda()
            one_dimension_result.cuda()

        SummResult.zero_()

        #here i extract boundaries from sample, format binares picture
        BinareFilterSample = boundaries_detect_laplacian(face_dataset[num_sample_in])/255
        if torch.cuda.is_available(): BinareFilterSample.cuda()

        #here i compare k features_pictures with layer on the sample picture with srtide 1 px
        k_range=feature_amount
        i_range=layers_by_wall-from_layer
        j_range=size_pic_X-feature_size_X
        for k in range(0,k_range):
            #print(k)
            for i in range(0,i_range):
                for j in range(0, j_range):
                    #here i calculate coordinates clice of sample to compare with feature map templets
                    #operand 'if' is used to decrease time calc, the k=0 it is a pure horizontal line on feature map, if at thix point (i,j) no horisontal line hence no another feature also
                    if (k==0) | (SummResult[0,i,j]>4):
                        x1=j
                        x2=j+feature_size_X
                        y1=size_pic_Y-feature_size_Y-(i+from_layer)
                        y2=size_pic_Y-(i+from_layer)
                        local_feature = BinareFilterSample[y1:y2, x1:x2]
                        #here i multiply pixel by pixel,this operation save only nonzero elements at both matrix, futher i summ all nonzero elements
                        SummResult[k,i,j] = torch.sum(local_feature*feature_map[k])+1
                        one_dimension_result[num_sample_in-num_sample_from, 0, k*i_range*j_range+i*j_range+j]=SummResult[k,i,j]
                        #if SummResult[k,i,j]>4: heating_map[y1:y2,x1:x2]=heating_map[y1:y2,x1:x2]+feature_map.long()[k]

    one_dimension_result=m_pool_1d(one_dimension_result)
    test='false'
    if test=='true':
        # here i show results of feature_map impacting
        fig = plt.figure()
        ax = plt.subplot(2, 1, 1)  # coordinates
        plt.tight_layout()
        ax.set_title('Sample boundaries #{}'+str(num_sample_in))
        ax.axis('off')
        plt.imshow(BinareFilterSample, 'gray')
        ax = plt.subplot(2, 1, 2)  # coordinates
        plt.tight_layout()
        ax.set_title('Heating map'+str(num_sample_in))
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
    #face_dataset = FramesDataset('./train/annotations.csv', './train')
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
    hidden_layer=1
    hidden_features=1
    sequence_len=100
    number_of_samples_lstm=120000
    first_sample_lstm=28*12000 #63 * 12000
    error=numpy.zeros(shape=(math.floor(number_of_samples_lstm/sequence_len)), dtype='float32')
    input = (capture_feature(face_dataset, feature_map, first_sample_lstm, first_sample_lstm+sequence_len, 0, 10)[1])
    print(input)
    target=torch.FloatTensor(math.floor(number_of_samples_lstm/sequence_len))

    # number of features input, number of features hidden layer ,2- number or recurent layers
    rnn = torch.nn.LSTM(input.data.shape[2], hidden_features, hidden_layer, dropout=0.1)
    print(input.data.shape)
    input_captured=Variable(torch.FloatTensor(math.floor(number_of_samples_lstm/sequence_len), sequence_len, 1, input.data.shape[2]))#tensor for repead using of captured feature
    optimizer = torch.optim.Adadelta(rnn.parameters(), lr=1)

    h0 = torch.autograd.Variable(torch.FloatTensor(sequence_len,1,hidden_features))
    h0.data[0,0,0]=0.01
    c0 = torch.autograd.Variable(torch.FloatTensor(sequence_len,1,hidden_features))
    c0.data[0, 0, 0] =0.01
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output)

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


    from_video='true'
    if from_video=='true':

        for sequence_num in range(0, math.floor((number_of_samples_lstm) / sequence_len)):
            sample_num = first_sample_lstm + sequence_num * sequence_len
            print(sample_num)
            target[sequence_num] = float(face_dataset[sample_num]['heat_transfer'])
            # print(sample_num)
        # normalization
        #target = target / torch.max(target)
        target = Variable(target)
        print('heat transfer headings are loaded')
        print(target)
        torch.save(target, '/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train_heatload.pt')


        samples_indexes = [i for i in range(0, math.floor((number_of_samples_lstm) / sequence_len))]  # A list contains all shuffle requires numbers
        shuffle(samples_indexes)

        #cycle by all samples
        for index, sequence_num in enumerate(samples_indexes):

            input = (capture_feature(face_dataset, feature_map, first_sample_lstm+sequence_num*sequence_len,first_sample_lstm+(sequence_num+1)*sequence_len, 0, 10)[1])
            input_captured[sequence_num] = input

            output, (hn, cn) = rnn(input,(h0,c0))

            w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
            w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

            loss = ((target[sequence_num])-10000*torch.sum(hn))
            error[sequence_num]=loss.data[0]

            loss.backward()
            optimizer.step()

            print(str(first_sample_lstm+sequence_num*sequence_len)+'-'+ str(first_sample_lstm+(sequence_num+1)*sequence_len)+' '+str("%.4f" %torch.sum(w_ii - u_ii).data[0]) + '  ' + str("%.4f" %torch.sum(w_if - u_if).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_ic - u_ic).data[0]) + '  ' + str("%.4f" %torch.sum(w_io - u_io).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_hi - u_hi).data[0]) + '  ' + str("%.4f" %torch.sum(w_hf - u_hf).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_hc - u_hc).data[0]) + '  ' + str("%.4f" %torch.sum(w_ho - u_ho).data[0]) + '  loss=' + str(
                "%.4f" %loss.data[0])+'  out'+str("%.4f" %torch.sum(hn).data[0])+'   '+ str(torch.sum(input).data[0]) + '   heat='+str(target[sequence_num].data[0]))

            optimizer.zero_grad()

            for hn_i in range(0,sequence_len):print(torch.mean(output[hn_i]).data[0])

        torch.save(input_captured.byte(), '/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train.pt')
        torch.save(target, '/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train_heatload.pt')
    else:
        input_captured=torch.load('/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train_seq_100.pt').float()
        target=torch.load('/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train_heatload_seq_100.pt')
        print('this tensor is loaded from file')
        print(input_captured.shape)

    print('normalization')
    for sequence_num in range(0, math.floor((number_of_samples_lstm) / sequence_len)):
        print(sequence_num)
        input_captured[sequence_num]=input_captured[sequence_num]/torch.max(input_captured[sequence_num])

    print('learning started')
        # repead cycle by all samples
    for repead in range(0,10):
        print('repeat'+str(repead))

        samples_indexes = [i for i in range(0, math.floor((number_of_samples_lstm) / sequence_len))]  # A list contains all shuffle requires numbers
        shuffle(samples_indexes)


        for index, sequence_num in enumerate(samples_indexes):
            print(sequence_num)
            input = input_captured[sequence_num]

            output, (hn, cn) = rnn(input, (h0, c0))

            w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
            w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

            loss = ((target[sequence_num])-torch.mean(hn))
            error[sequence_num] = loss.data[0]

            loss.backward()
            optimizer.step()
            print(str(first_sample_lstm + sequence_num * sequence_len) + '-' + str(
                first_sample_lstm + (sequence_num + 1) * sequence_len) + ' ' + str(
                "%.4f" % torch.sum(w_ii - u_ii).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_if - u_if).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_ic - u_ic).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_io - u_io).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_hi - u_hi).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_hf - u_hf).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_hc - u_hc).data[0]) + '  ' + str(
                "%.4f" % torch.sum(w_ho - u_ho).data[0]) + '  loss=' + str(
                "%.4f" % loss.data[0]) + '  out' + str("%.4f" % torch.mean(hn).data[0]) + '   '+ str(torch.sum(input).data[0]) + '   heat='+str(target[sequence_num].data[0]*75000))

            #for hn_i in range(0,sequence_len):print(torch.mean(output[hn_i]).data[0])

            u_ii = w_ii.clone()
            u_if = w_if.clone()
            u_ic = w_ic.clone()
            u_io = w_io.clone()
            u_hi = w_hi.clone()
            u_hf = w_hf.clone()
            u_hc = w_hc.clone()
            u_ho = w_ho.clone()
            output_prev=output.clone()
            optimizer.zero_grad()

    print(u_ii)
    print(u_if)
    print(u_ic)
    print(u_io)
    print('max io ' + str(torch.max(u_io)) + 'min io ' + str(torch.min(u_io)))
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