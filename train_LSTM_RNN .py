import matplotlib.pyplot as plt
import torch
import numpy
from torch.autograd import Variable
import numpy as np
import math
import datetime
now = datetime.datetime.now()
import os

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from picture_transformation import init_edge_feature_map_5x5
from random import shuffle

def captured_decomposer(input,feature_map,time_step_speed):
    #captured features are stored in memory in optimal format(as number of founded feature), decompozer create array named input (spare matrix) for sequensce
    """
    :param input: array of captured feature numbers for every x coordinat
    :param feature_map: array of with feature map 1 dimension-number of temple, 2-3 dimentions size temple of picture
    :param time_step_speed: number of consistently frame to create dictionary
    """
    feature_amount = feature_map.shape[0]
    number_steps_signature=input.data.shape[2]
    j_range=input.data.shape[0]

    one_dimension_result = torch.FloatTensor(input.data.shape[0], input.data.shape[1],(feature_amount ** time_step_speed) * number_steps_signature).zero_()

    for j in range(0,j_range):
        for signature_step in range(0,number_steps_signature):
            one_dimension_result[j, 0, input.data[j,0,signature_step] + (feature_amount ** time_step_speed) * signature_step] = 1



    return Variable(one_dimension_result)

def capture_feature(face_dataset, feature_map, num_sample_from, num_sample_to, from_layer, layers_by_wall, time_speed):
    """
    :param face_dataset: the video dataset like a group of a pictures
    :param feature_map: array of with feature map 1 dimension-number of temple, 2-3 dimentions size temple of picture
    :param num_samples_from: number of picture to calculation sequence of pictures
    :param num_samples_to: number of picture to calculation sequence of pictures
    :param layers_by_wall: number of layers by wall where i search features
    :param from_layer: number of layers from edge of picture by Y
    :param time_speed: number of samples to calculate local speed of edges cheinging
    """
    #final results matrix will has lower second two dimension(7x340) than initial pictures (48x340) because we use only part by the wall, first dimension is equal feature dimension of feature_map
    m_pool_1d = torch.nn.MaxPool1d(20)
    size_pic_X = face_dataset[num_sample_from]['frame'].shape[1]
    size_pic_Y = face_dataset[num_sample_from]['frame'].shape[0]
    feature_size_X = feature_map.shape[2]
    feature_size_Y = feature_map.shape[1]
    feature_amount = feature_map.shape[0]
    k_range = feature_amount
    i_range = layers_by_wall - from_layer
    j_range = size_pic_X - feature_size_X
    number_steps_signature=int((num_sample_to-num_sample_from)/time_speed)


    SummResult = torch.FloatTensor(feature_amount, layers_by_wall - from_layer, size_pic_X).zero_()
    one_dimension_result = torch.FloatTensor(size_pic_X - feature_size_X, 1, (feature_amount**time_speed)*number_steps_signature).zero_()
    heating_map_numpy = np.zeros(shape=(size_pic_Y, size_pic_X), dtype='int64')
    heating_map = torch.from_numpy(heating_map_numpy)
    statistic_map = np.zeros(shape=(j_range), dtype='int64')
    max_local = torch.FloatTensor(time_speed).zero_()
    max_local_number=torch.IntTensor(time_speed).zero_()
    captured_features = torch.IntTensor(size_pic_X - feature_size_X, 1, number_steps_signature).zero_()# (X, 1, nonzero elements number for every time step)

    # here i find boundaries for 3 pictures
    t=boundaries_detect_laplacian(face_dataset[num_sample_from])/255
    BinareFilterSample=torch.ByteTensor(number_steps_signature*time_speed,t.shape[0],t.shape[1])
    BinareFilterSample[0]=t
    for num_sample_in in range(num_sample_from+1, num_sample_to):
        # here i extract boundaries from sample, format binares picture
        BinareFilterSample[num_sample_in-num_sample_from] = boundaries_detect_laplacian(face_dataset[num_sample_in]) / 255


    if torch.cuda.is_available():
        SummResult.cuda()
        one_dimension_result.cuda()
        BinareFilterSample.cuda()
    SummResult.zero_()



    for j in range(0, j_range):


        for signature_step in range(0,number_steps_signature):

            max_local.zero_()
            max_local_number.zero_()

            num_sample_local_from=num_sample_from+signature_step*time_speed
            num_sample_local_to=num_sample_from+signature_step*time_speed+time_speed

            for num_sample_in in range(num_sample_local_from, num_sample_local_to):


                for i in range(0,i_range):
                    # here i compare k features_pictures with layer on the sample picture with srtide 1 py

                    for k in range(0, k_range):

                        #here i calculate coordinates clice of sample to compare with feature map templets
                        #operand 'if' is used to decrease time calc, the k=0 it is a pure horizontal line on feature map, if at thix point (i,j) no horisontal line hence no another feature also
                        if (k==0) | (SummResult[0,i,j]>4):
                            x1=j
                            x2=j+feature_size_X
                            y1=size_pic_Y-feature_size_Y-(i+from_layer)
                            y2=size_pic_Y-(i+from_layer)
                            local_feature = BinareFilterSample[num_sample_in - num_sample_from][y1:y2, x1:x2]
                            #here i multiply pixel by pixel,this operation save only nonzero elements at both matrix, futher i summ all nonzero elements
                            SummResult[k,i,j] = torch.sum(local_feature*feature_map[k])+1
                            #here i save the number of max coincide element by Y axis
                            if(SummResult[k,i,j]>max_local[num_sample_in-num_sample_local_from]):
                                max_local[num_sample_in - num_sample_local_from]=SummResult[k,i,j]
                                max_local_number[num_sample_in - num_sample_local_from]=k

            #here i calculate adress of nonzero element #k * k_range* k_range + i * k_range + j
            nonzero_number=0
            for num_sample_in in range(num_sample_local_from, num_sample_local_to):
                nonzero_number=max_local_number[num_sample_in - num_sample_local_from]*(k_range**(num_sample_in - num_sample_local_from))+nonzero_number


            if nonzero_number>0:
                one_dimension_result[j, 0, nonzero_number+(feature_amount**time_step_speed)*signature_step] = 1
                #print('x= '+str(j)+'  nonzero_number= '+ str(nonzero_number+(feature_amount**time_step_speed)*signature_step))
                statistic_map[j]=nonzero_number
                captured_features[j, 0,  signature_step] = nonzero_number# here i save captured number to reuse it after saving to file (nonzero number this is combination of two feature form two samples in the x coordinate


    #one_dimension_result=m_pool_1d(one_dimension_result)
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

    return SummResult, Variable(one_dimension_result), statistic_map, Variable(captured_features)

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


    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset('file:///media/alexander/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/alexander/Files/@Machine/Github/Boiler/train')
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

    #model time step (sequense is a pixels during X-axis on 3 time step frames)
    hidden_layer=2
    hidden_features=1
    time_step_speed=2# number of time steps to put into vocabulary(actially it is a function of the speed of the boundaries of a buble)
    time_step_signature=30# number of samples added to signature of state of boiling


    number_of_samples_lstm=21*12000
    first_sample_lstm=28*12000 #63 * 12000

    number_of_sequences=int(math.floor(number_of_samples_lstm/(time_step_speed*time_step_signature)))

    SummResult, input, statistic_map, capture_example = (capture_feature(face_dataset, feature_map, first_sample_lstm, first_sample_lstm+time_step_speed*time_step_signature, 0, 10,time_step_speed))

    sequence_len = input.data.shape[0]# number of pixelx by X of picture
    error=numpy.zeros(shape=(number_of_sequences), dtype='float32')
    error_by_heat=numpy.zeros(shape=(number_of_sequences), dtype='float32')
    statistic_maps=np.zeros(shape=(number_of_sequences, statistic_map.shape[0]), dtype='int64')
    heat_statistics=np.zeros(shape=(number_of_sequences, statistic_map.shape[0]), dtype='int64')

    print('input 0')

    target=torch.FloatTensor(number_of_sequences)

    # number of features input, number of features hidden layer ,2- number or recurent layers
    rnn = torch.nn.LSTM(input.data.shape[2], hidden_features, hidden_layer, dropout=0.1)
    ln1=torch.nn.Linear(input.data.shape[2],100)
    ln2 = torch.nn.Linear(100, hidden_features)

    print(input.data.shape)
    input_captured=Variable(torch.IntTensor(number_of_sequences, capture_example.data.shape[0], capture_example.data.shape[1], capture_example.data.shape[2]))#tensor for repead using of captured feature
    optimizer1 = torch.optim.Adadelta(ln1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adadelta(ln2.parameters(), lr=0.001)


    h0 = torch.autograd.Variable(torch.FloatTensor(sequence_len,1,hidden_features))
    h0.data[0,0,0]=0.01
    c0 = torch.autograd.Variable(torch.FloatTensor(sequence_len,1,hidden_features))
    c0.data[0, 0, 0] =0.01
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output)

    w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
    #first layer LSTM
    u_ii0 = w_ii.clone()
    u_if0 = w_if.clone()
    u_ic0 = w_ic.clone()
    u_io0 = w_io.clone()
    u_hi0 = w_hi.clone()
    u_hf0 = w_hf.clone()
    u_hc0 = w_hc.clone()
    u_ho0 = w_ho.clone()
    #second layer LSTM
    u_ii = w_ii.clone()
    u_if = w_if.clone()
    u_ic = w_ic.clone()
    u_io = w_io.clone()
    u_hi = w_hi.clone()
    u_hf = w_hf.clone()
    u_hc = w_hc.clone()
    u_ho = w_ho.clone()


    from_video='false'
    if from_video=='true':

        #here i create a tensor with heatload for a loss function
        for sequence_num in range(0, number_of_sequences):
            sample_num = first_sample_lstm + sequence_num * time_step_speed*time_step_signature
            print(sample_num)
            target[sequence_num] = float(face_dataset[sample_num]['heat_transfer'])/100000
            # print(sample_num)
        # normalization
        #target = target / torch.max(target)
        target = Variable(target)
        print('heat transfer headings are loaded')
        print(target)
        torch.save(target, '/media/aleksandr/Files/@Machine/Github/Boiler/boiling_train_heatload_3time step.pt')


        samples_indexes = [i for i in range(0, number_of_sequences)]  # A list contains all shuffle requires numbers
        shuffle(samples_indexes)
        sequence_num=samples_indexes[0]
        SummResult, input, statistic_map, input_captured[0] = (capture_feature(face_dataset, feature_map, first_sample_lstm+sequence_num*time_step_speed*time_step_signature,first_sample_lstm+(sequence_num+1)*time_step_speed*time_step_signature, 0, 10,time_step_speed))

        #cycle by all samples
        for index, sequence_num in enumerate(samples_indexes):
            #sequence_num = samples_indexes[0]
            SummResult, input, statistic_map, input_captured[sequence_num] = (capture_feature(face_dataset, feature_map, first_sample_lstm+sequence_num*time_step_speed*time_step_signature,first_sample_lstm+(sequence_num+1)*time_step_speed*time_step_signature, 0, 10,time_step_speed))
            #input_captured[sequence_num] = input

            statistic_maps[sequence_num]=statistic_map

            output, (hn, cn) = rnn(input,(h0,c0))

            w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
            w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)
            b_hi, b_hf, b_hc, b_ho=rnn.bias_hh_l0.chunk(4, 0)

            loss = ((target[sequence_num])- torch.max(hn*w_ho+b_ho))**2
            error[index]=loss.data[0]

            loss.backward()
            optimizer1.step()

            #print(torch.nn.register_backward_hook(rnn))

            print(str(index)+'  '+str(first_sample_lstm+sequence_num*time_step_speed*time_step_signature)+'-'+ str(first_sample_lstm+(sequence_num+1)*time_step_speed*time_step_signature)+' '+str("%.4f" %torch.sum(w_ii - u_ii).data[0]) + '  ' + str("%.4f" %torch.sum(w_if - u_if).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_ic - u_ic).data[0]) + '  ' + str("%.4f" %torch.sum(w_io - u_io).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_hi - u_hi).data[0]) + '  ' + str("%.4f" %torch.sum(w_hf - u_hf).data[0]) + '  ' + str(
                "%.4f" %torch.sum(w_hc - u_hc).data[0]) + '  ' + str("%.4f" %torch.sum(w_ho - u_ho).data[0]) + '  loss=' + str(
                "%.4f" %loss.data[0])+'  out'+str("%.4f" %torch.max(hn).data[0])+'   '+ str(torch.sum(input).data[0]) + '   heat='+str(100000*target[sequence_num].data[0]))

            # here i show results of feature_map impacting
            #plt.clf()
            #plt.title('feature statistics')
            #for i_state in range(0, index):
            #    print(i_state)
            #    i_state_seq_num=samples_indexes[i_state]
            #    heat_statistics[i_state_seq_num] = 100000 * (target[i_state_seq_num])+5000*random.random()
            #    plt.plot(heat_statistics[i_state_seq_num], statistic_maps[i_state_seq_num], 'bo', label='features')
            #plt.ylabel('Feature')
            #plt.xlabel('Heatload')
            #plt.show()

            if index==4000*int(index/4000):
                plt.clf()
                plt.axes([0.3, 0.3, 0.5, 0.5])
                # plt.title ('iteration error for ' + str(5) +' max eigenvalues and eigenvectors')
                plt.plot(error[0:index], 'k:', label='1')
                plt.xlabel('Iteration')
                plt.ylabel('error')
                plt.legend()
                plt.show()


            u_ii = w_ii.clone()
            u_if = w_if.clone()
            u_ic = w_ic.clone()
            u_io = w_io.clone()
            u_hi = w_hi.clone()
            u_hf = w_hf.clone()
            u_hc = w_hc.clone()
            u_ho = w_ho.clone()

            optimizer1.zero_grad()

            #for hn_i in range(0,sequence_len):print(torch.mean(output[hn_i]).data[0])

        torch.save(input_captured, '/media/alexander/Files/@Machine/Github/Boiler/boiling_train_2time_step.pt')
        torch.save(target, '/media/alexander/Files/@Machine/Github/Boiler/boiling_train_heatload_2time_step.pt')
    else:
        input_captured=torch.load('/media/alexander/Files/@Machine/Github/Boiler/boiling_train_2time_step.pt')
        target=torch.load('/media/alexander/Files/@Machine/Github/Boiler/boiling_train_heatload_2time_step.pt')
        print('this tensor is loaded from file')
        print(input_captured.shape)
        print('heat load')
        print(target.shape)
        print('load 1')
        print(target)


    number_of_seq_together=10 #number of sequences by time_step_signature added to one sequence for reusing code
    number_of_new_sequences=int(math.floor(number_of_sequences/number_of_seq_together))#number of sequrnces after adding seq to one
    new_input_captured=Variable(torch.IntTensor(number_of_new_sequences, input_captured.data.shape[1], input_captured.data.shape[2], input_captured.data.shape[3]*number_of_seq_together))#tensor for repead using of captured feature
    new_target=Variable(torch.FloatTensor(number_of_new_sequences))
    error=numpy.zeros(shape=(number_of_new_sequences), dtype='float32')
    error_by_heat=numpy.zeros(shape=(number_of_new_sequences), dtype='float32')
    zero_load_repeat=5

    print('changing size')
    #here i change size of one sequence
    for new_sequences in range(0,number_of_new_sequences):
        new_target[new_sequences]=target[new_sequences*number_of_seq_together]
        print('new_sequence'+str(new_sequences))
        seq_from=new_sequences*number_of_seq_together
        seq_to=(new_sequences+1)*number_of_seq_together
        for sequence in range (seq_from, seq_to):
            for j in range (0,new_input_captured.data.shape[1]):
                for feature in range(0, input_captured.data.shape[3]):
                    a=input_captured.data[sequence,j,0,feature]
                    new_input_captured.data[new_sequences,j,0,(sequence-seq_from)*input_captured.data.shape[3]+feature]=a

    print(new_input_captured[10,0,0])
    print(input_captured[100,0,0])
    input_captured=new_input_captured
    target=new_target
    number_of_sequences=number_of_new_sequences

    input = captured_decomposer(input_captured[0], feature_map, time_step_speed)

    rnn = torch.nn.LSTM(input.data.shape[2], hidden_features, hidden_layer, dropout=0.1)
    ln1=torch.nn.Linear(input.data.shape[2],100)
    ln2=torch.nn.Linear(100,1)
    optimizer1 = torch.optim.Adadelta(ln1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adadelta(ln2.parameters(), lr=0.001)
    optimizerLSTM=torch.optim.Adadelta(rnn.parameters(), lr=0.001)



    steps_to_print=number_of_sequences-1
    print('learning started')
        # repead cycle by all samples
    for era in range(0,10):
        print('learning era'+str(era+1))

        samples_indexes = [i for i in range(0, number_of_sequences)]  # A list contains all shuffled requires numbers
        shuffle(samples_indexes)


        for index, sequence_num in enumerate(samples_indexes):

            input = captured_decomposer(input_captured[sequence_num], feature_map, time_step_speed)

            output = ln1(input)
            output = ln2(output)

            loss = ((target[sequence_num]) - torch.sum(output)) ** 2
            error[index] = loss.data[0]
            error_by_heat[sequence_num] = ((target[sequence_num]) - torch.sum(output))

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()


            #here i repeat passing through zero load elements, to increase their weight at the all data
            if target.data[sequence_num]==0 :
                for zero_repeat in range(0,zero_load_repeat):

                    output = ln1(input)
                    output = ln2(output)
                    loss = ((target[sequence_num]) - torch.sum(output)) ** 2
                    error[index] = loss.data[0]
                    error_by_heat[sequence_num] = ((target[sequence_num]) - torch.sum(output))

                    loss.backward()
                    optimizer1.step()
                    optimizer2.step()

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    print(str(index) + '  ' + str(
                        first_sample_lstm + sequence_num * time_step_speed * time_step_signature) + '-' + str(
                        first_sample_lstm + (
                                    sequence_num + 1) * time_step_speed * time_step_signature) + ' ' + '  out ' + str(
                        "%.4f" % torch.sum(output).data[0]) + '   ' + str(
                        torch.sum(input).data[0]) + '   heat=' + str(
                        "%.1f" % (100000 * target[sequence_num].data[0])) + '  loss=' + str("%.4f" % loss.data[0]))



            print(str(index) + '  ' + str(
                first_sample_lstm + sequence_num * time_step_speed * time_step_signature) + '-' + str(
                first_sample_lstm + (sequence_num + 1) * time_step_speed * time_step_signature) + ' ' + '  out ' + str("%.4f" % torch.sum(output).data[0]) + '   ' + str(
                torch.sum(input).data[0]) + '   heat=' + str("%.1f" %(100000 * target[sequence_num].data[0]))+ '  loss=' + str(
                "%.4f" % loss.data[0]))

            if index==steps_to_print*int(index/steps_to_print):
                plt.clf()
                plt.axes([0.3, 0.3, 0.5, 0.5])
                plt.title ('iteration loss(index),300 times 2 layer Linear,*5 zero load, era'+str(era+1))
                plt.plot(error, 'k:', label='1')
                plt.xlabel('Iteration')
                plt.ylabel('loss')
                plt.legend()
                basePath = os.path.dirname(os.path.abspath(__file__))
                results_dir = basePath+ '/Models/LSTM/08_04_18_X-Time/'
                sample_file_name = 'x5 Error_linear2layer_' + str(steps_to_print) + '_steps_era_'+str(era)+'.png'
                plt.savefig(results_dir + sample_file_name)

                plt.clf()
                plt.axes([0.3, 0.3, 0.5, 0.5])
                plt.title ('iteration error (heat),300 times 2 layer Linear,*5 zero load, era'+str(era+1))
                plt.plot(error_by_heat, 'k:', label='1')
                plt.xlabel('Heat load')
                plt.ylabel('error')
                plt.legend()
                sample_file_name = 'x5 Error_linear_arranged_by_load' + str(steps_to_print) + '_steps_era_'+str(era)+'.png'
                plt.savefig(results_dir + sample_file_name)




             # ... after training, save your model
        #rnn.save_state_dict('LSTM_'+str(repead+1)+'_learning 31_03_18.pt')

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
    plt.title('iteration error by index, learning era' + str(era + 1))
    plt.plot(error, 'k:', label='1')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    plt.clf()
    plt.axes([0.3, 0.3, 0.5, 0.5])
    plt.title('iteration error heat loads, learning era' + str(era + 1))
    plt.plot(error_by_heat, 'k:', label='1')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.legend()
    plt.show()

