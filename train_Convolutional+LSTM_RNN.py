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
from random import shuffle
import visualize


def hookFunc1(module, gradInput, gradOutput):
    output = 'conv1'
    for v in gradInput:
        if v is None:
            output=output + ' none'
        else:
            output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def hookFunc2(module, gradInput, gradOutput):
    output = 'conv2'
    for v in gradInput:
        output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def hookFunc3(module, gradInput, gradOutput):
    output = 'conv3'
    for v in gradInput:
        output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def hookFunc4(module, gradInput, gradOutput):
    output = 'conv4'
    for v in gradInput:
        output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def hookFunc5(module, gradInput, gradOutput):
    output = 'conv5'
    for v in gradInput:
        output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def hookLSTM(module, gradInput, gradOutput):
    output = 'LSTM'
    for v in gradInput:
        output = output + ' ' + str(torch.max(torch.abs(v)).data[0])
    print(output)


def test_dataset(face_dataset,from_frame):
    # visualize some data
    sample = face_dataset[1]
    print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    for j in range(0, 1):
        for i in range(1, 10):
            # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
            SummResult = face_dataset[from_frame + i + j * 13]['frame']
            # here i show results
            ax = plt.subplot(11 // 3 + 1, 3, i)  # coordinates
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            # show the statistic matrix
            plt.imshow(SummResult, 'gray')

        plt.show()


if __name__ == "__main__":

    print('Cuda available?')
    print(torch.cuda.is_available())
    print('videocard')
    print(torch.cuda.device_count())

    basePath=os.path.dirname(os.path.abspath(__file__))

    video_length = 12000
    number_of_samples_lstm = 50 * video_length
    first_sample_lstm = 26 * video_length

    number_of_samples_lstm_validation = 20 * video_length
    first_sample_lstm_validation =77 * video_length

    #here i load the video dataset like a group of a pictures
    run_key='verification'
    face_dataset = FramesDataset(basePath+'/train/annotations.csv',basePath+ '/train')
    test_dataset(face_dataset,first_sample_lstm_validation)
    zero_load_repeat=0
    number_of_farme_per_batch=30
    first_layer_features_number=76





    #print('frame type')
    #print(face_dataset[first_sample_lstm]['frame'])
    #print(face_dataset[first_sample_lstm]['frame'].shape)
    #print(type(face_dataset[first_sample_lstm]['frame'][0,0]))

    number_of_sequences=int(math.floor(number_of_samples_lstm/number_of_farme_per_batch))
    number_of_sequences_validation=int(math.floor(number_of_samples_lstm_validation/number_of_farme_per_batch))

    #model Convolutional
    conv_layer_1=torch.nn.Conv2d(1,first_layer_features_number,7) #76-even number of significant 2d features
    max_pool_layer_1=torch.nn.MaxPool2d(face_dataset[first_sample_lstm]['frame'].shape[0]+1-7) #convert features tensors to 1d tensors, with kernel size equal hight of picture
    conv_layer_2=torch.nn.Conv1d(first_layer_features_number,first_layer_features_number*2,2) #
    max_pool_layer_2 = torch.nn.MaxPool1d(2)
    conv_layer_3=torch.nn.Conv1d(first_layer_features_number*2,first_layer_features_number*2*2,2) #
    max_pool_layer_3 = torch.nn.MaxPool1d(2)
    conv_layer_4=torch.nn.Conv1d(first_layer_features_number*2*2,first_layer_features_number*2*2*2,2) #
    max_pool_layer_4 = torch.nn.MaxPool1d(2)
    conv_layer_5=torch.nn.Conv1d(first_layer_features_number*2*2*2,first_layer_features_number*2*2*2*2,2) #
    max_pool_layer_5 = torch.nn.MaxPool1d(2)

    conv_layer_1.cuda()
    max_pool_layer_1.cuda()
    conv_layer_2.cuda()
    max_pool_layer_2.cuda()
    conv_layer_3.cuda()
    max_pool_layer_3.cuda()
    conv_layer_4.cuda()
    max_pool_layer_4.cuda()
    conv_layer_5.cuda()
    max_pool_layer_5.cuda()


    input=Variable(torch.cuda.FloatTensor(number_of_farme_per_batch,1,face_dataset[first_sample_lstm]['frame'].shape[0],face_dataset[first_sample_lstm]['frame'].shape[1]).zero_())
    #input_captured=torch.FloatTensor(number_of_sequences,number_of_farme_per_batch,1,face_dataset[first_sample_lstm]['frame'].shape[0],face_dataset[first_sample_lstm]['frame'].shape[1]).zero_()



    #print('input_captured')
    #print(input_captured.data.shape)
    #print('is cuda? '+str(input_captured.is_cuda))

    for i in range(0, number_of_farme_per_batch):
        input.data[i,0]=torch.from_numpy(face_dataset[first_sample_lstm+i]['frame'])


    print('input')
    print(input)


    output=conv_layer_1(input)
    print('conv_layer_1')
    print(output)


    output=max_pool_layer_1(output)
    print('max_pool_layer_1')
    print(output)


    output=torch.squeeze(output,2)
    print('squeeze 1 dimension')
    print(output)


    output=conv_layer_2 (output)
    print('conv_layer_2')
    print(output)


    output=max_pool_layer_2(output)
    print('max_pool_layer_2')
    print(output)


    output=conv_layer_3 (output)
    print('conv_layer_3')
    print(output)


    output=max_pool_layer_3(output)
    print('max_pool_layer_3')
    print(output)

    output=conv_layer_4 (output)
    print('conv_layer_4')
    print(output)

    output=max_pool_layer_4(output)
    print('max_pool_layer_4')
    print(output)


    for param in max_pool_layer_5.parameters():
        print(type(param.data), param.size())


    output=torch.squeeze(output,2)
    print('squeeze 2 dimension')
    print(output)

    output=torch.unsqueeze(output,1)
    print('unsquese')
    print(output)

    output=(output-torch.mean(output))/torch.max(torch.abs(output))

    print('mean output conv' + str(torch.mean(output)))

    #model LSTM
    hidden_layer=1
    hidden_features=1


    # here i init NN
    # The structure of input tensor LSTM fo picture recognition (seq-lenght, batch, input_size)
    # 1 argument - is equal to number of pictures- for every picture is calculated h-t and putted to output
    # 2 argument - is equal a number of batch - better to use 1
    # 3 argument - is one dimensional array contains all features from all part of picture- it is require convert 3 dimensional array to 1 dimension and put to this cell.
    # The structure of paramenters LSTM(lenght of array with features=big value, number of heatures in output can be lower and higer than in input- how mach i want, number of layer in recurent model)

    rnn = torch.nn.LSTM(output.data.shape[2], hidden_features, hidden_layer, dropout=0.01)







    if torch.cuda.is_available()==True:
        rnn.cuda()



    optimizerLSTM=torch.optim.Adadelta([
                                        {'params': rnn.parameters()},
                                        {'params': conv_layer_1.parameters()},
                                        {'params': conv_layer_2.parameters()},
                                        {'params': conv_layer_3.parameters()},
                                        {'params': conv_layer_4.parameters()}
                                        ], lr=0.1)


    sequence_len = output.data.shape[0]
    h0 = torch.autograd.Variable(torch.cuda.FloatTensor(hidden_layer,1,hidden_features)).fill_(0.01)
    c0 = torch.autograd.Variable(torch.cuda.FloatTensor(hidden_layer,1,hidden_features)).fill_(0.01)

    # load pretrained model if it is required
    #[rnn, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5] = torch.load('№7_model_01.pt')
    #rnn.flatten_parameters()

    # show first layer weigts
    weignt_1conv, bias_1conv = conv_layer_1.parameters()
    visualize.show_weights(weignt_1conv.data)

    if torch.cuda.is_available()==True:
        print('c0 inside cuda ' + str(c0.is_cuda))
        print('h0 inside cuda ' + str(h0.is_cuda))

    output, (hn, cn) = rnn(output, (h0, c0))
    print(output)

    #rnn.register_backward_hook(hookLSTM)
    #conv_layer_1.register_backward_hook(hookFunc1)
    #conv_layer_2.register_backward_hook(hookFunc2)
    #conv_layer_3.register_backward_hook(hookFunc3)
    #conv_layer_4.register_backward_hook(hookFunc4)
    #conv_layer_5.register_backward_hook(hookFunc5)

    w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)
    w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

    print('w_ii inside cuda' + str(w_ii.is_cuda))

    print(w_ii)
    print(w_if)
    print(w_ic)
    print(w_io)

    print(w_hi)
    print(w_hf)
    print(w_hc)
    print(w_ho)


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

    error=torch.cuda.FloatTensor(number_of_sequences)
    error_by_heat=torch.cuda.FloatTensor(number_of_sequences)
    heat_predicted = torch.cuda.FloatTensor(number_of_sequences)
    target = torch.cuda.FloatTensor(number_of_sequences)


    print('target generation (heat load)')
    for sequence_num in range(0, number_of_sequences):
        sample_num = first_sample_lstm + sequence_num*number_of_farme_per_batch
        print(sample_num)
        target[sequence_num] = float(face_dataset[sample_num]['heat_transfer']) / 100000

        #put all video to one tensor (GPU or not)
        #for i in range(0, number_of_farme_per_batch):
        #    input_captured[sequence_num,i, 0] = torch.from_numpy(face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'])

        # print(sample_num)
    target = Variable(target)

    #input_captured_pinned = torch.Tensor.pin_memory(input_captured)



    error_validation=torch.cuda.FloatTensor(number_of_sequences)
    error_by_heat_validation=torch.cuda.FloatTensor(number_of_sequences)
    heat_predicted_validation = torch.cuda.FloatTensor(number_of_sequences)
    target_validation = torch.cuda.FloatTensor(number_of_sequences)

    print('target validation generation (heat load)')
    for sequence_num in range(0, number_of_sequences_validation):
        sample_num = first_sample_lstm_validation + sequence_num * number_of_farme_per_batch
        print(sample_num)
        target_validation[sequence_num] = float(face_dataset[sample_num]['heat_transfer']) / 100000

    target_validation = Variable(target_validation)


    epoch_number=90
    train_vs_epoch=torch.cuda.FloatTensor(epoch_number).zero_()
    validation_vs_epoch=torch.cuda.FloatTensor(epoch_number).zero_()


    steps_to_print=number_of_sequences-1
    print('train started Convolution+LSTM')
        # repead cycle by all samples
    for epoch in range(0,epoch_number):
        print('learning epoch'+str(epoch+1))

        samples_indexes = [i for i in range(0, number_of_sequences)]  # A list contains all shuffled requires numbers
        shuffle(samples_indexes)


        for index, sequence_num in enumerate(samples_indexes):

            #print(str(index)+' from '+ str(number_of_sequences))

            for i in range(0, number_of_farme_per_batch):
                input.data[i, 0] = torch.from_numpy(face_dataset[first_sample_lstm + sequence_num*number_of_farme_per_batch +i]['frame'])
            #input=Variable(input_captured_pinned[sequence_num])

            output = conv_layer_1(input)
            output = max_pool_layer_1(output)
            output = torch.squeeze(output, 2)
            output = conv_layer_2(output)
            output = max_pool_layer_2(output)
            output = conv_layer_3(output)
            output = max_pool_layer_3(output)
            output = conv_layer_4(output)
            output = max_pool_layer_4(output)
            output = torch.squeeze(output, 2)
            output = torch.unsqueeze(output, 1)
            output =0.01* (output - torch.mean(output)) / torch.max(torch.abs(output))

            output, (hn, cn) = rnn(output, (h0, c0))

            w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)

            w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

            b_hi, b_hf, b_hc, b_ho = rnn.bias_hh_l0.chunk(4, 0)

            #print(str(w_ho.data[0,0])+'  '+str(b_ho.data[0])+'  '+ str(hn.data[0,0,0])+ '  '+ str(target[sequence_num].data[0]))


            loss = ((target[sequence_num]) - (hn * w_ho + b_ho)) ** 2

            error[index] = loss.data[0,0,0]

            error_by_heat[sequence_num] = ((target[sequence_num]) - (hn * w_ho + b_ho)).data[0,0,0]

            heat_predicted[sequence_num]=torch.max((hn * w_ho + b_ho)).data[0]

            loss.backward()

            optimizerLSTM.step()

            optimizerLSTM.zero_grad()


            if index==100*int(index/100): print(index)

            #here i repeat passing through zero load elements, to increase their weight at the all data

            if target.data[sequence_num]==0 :

                for zero_repeat in range(0,zero_load_repeat):

                    output = conv_layer_1(input)
                    output = max_pool_layer_1(output)
                    output = torch.squeeze(output, 2)
                    output = conv_layer_2(output)
                    output = max_pool_layer_2(output)
                    output = conv_layer_3(output)
                    output = max_pool_layer_3(output)
                    output = conv_layer_4(output)
                    output = max_pool_layer_4(output)
                    output = torch.squeeze(output, 2)
                    output = torch.unsqueeze(output, 1)
                    output = 0.01 * (output - torch.mean(output)) / torch.max(torch.abs(output))

                    output, (hn, cn) = rnn(output, (h0, c0))

                    w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)

                    w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

                    b_hi, b_hf, b_hc, b_ho = rnn.bias_hh_l0.chunk(4, 0)

                    loss = ((target[sequence_num]) - (hn * w_ho + b_ho)) ** 2

                    error[index] = loss.data[0, 0, 0]

                    error_by_heat[sequence_num] = ((target[sequence_num]) - (hn * w_ho + b_ho)).data[0, 0, 0]

                    heat_predicted[sequence_num] = torch.max((hn * w_ho + b_ho)).data[0]

                    loss.backward()

                    optimizerLSTM.step()

                    optimizerLSTM.zero_grad()


        #visualize.show_loss(index, w_ii - u_ii, w_if - u_if, w_ic - u_ic, w_io - u_io, w_hi - u_hi,
        #                    w_hf - u_hf, w_hc - u_hc, w_ho - u_ho, sequence_num, first_sample_lstm,
        #                    3, 300, loss, hn, error_by_heat.cpu().numpy(), target)




        print('validation epoch' + str(epoch + 1))

        for sequence_num in range(0,number_of_sequences_validation):

            #print(str(index)+' from '+ str(number_of_sequences))

            for i in range(0, number_of_farme_per_batch):
                input.data[i, 0] = torch.from_numpy(face_dataset[first_sample_lstm_validation + sequence_num*number_of_farme_per_batch +i]['frame'])

            output = conv_layer_1(input)
            output = max_pool_layer_1(output)
            output = torch.squeeze(output, 2)
            output = conv_layer_2(output)
            output = max_pool_layer_2(output)
            output = conv_layer_3(output)
            output = max_pool_layer_3(output)
            output = conv_layer_4(output)
            output = max_pool_layer_4(output)
            output = torch.squeeze(output, 2)
            output = torch.unsqueeze(output, 1)
            output =0.01* (output - torch.mean(output)) / torch.max(torch.abs(output))

            output, (hn, cn) = rnn(output, (h0, c0))

            w_ii, w_if, w_ic, w_io = rnn.weight_ih_l0.chunk(4, 0)

            w_hi, w_hf, w_hc, w_ho = rnn.weight_hh_l0.chunk(4, 0)

            b_hi, b_hf, b_hc, b_ho = rnn.bias_hh_l0.chunk(4, 0)

            loss = ((target_validation[sequence_num]) - (hn * w_ho + b_ho)) ** 2

            error_validation[sequence_num] = loss.data[0,0,0]

            error_by_heat_validation[sequence_num] = ((target_validation[sequence_num]) - (hn * w_ho + b_ho)).data[0,0,0]

            heat_predicted_validation[sequence_num]=torch.max((hn * w_ho + b_ho)).data[0]

            if index == 100 * int(index / 100): print(index)

        visualize.save_some_epoch_data(index, number_of_sequences-1, epoch, basePath, '/Models/LSTM/05_06_18_X-Time_N7/', 'Error_Conv+LSTM_N7_03', error_validation.cpu().numpy(), error_by_heat_validation.cpu().numpy(), run_key,'Conv 5 + LSTM 2, *5 zero load,')

        #here i create figure with the history of training and validation

        train_vs_epoch[epoch] = torch.mean(torch.abs(error_by_heat))

        validation_vs_epoch[epoch]=torch.mean(torch.abs(error_by_heat_validation))

        visualize.save_train_validation_picture(train_vs_epoch.cpu().numpy()[0:epoch],validation_vs_epoch.cpu().numpy()[0:epoch], basePath, '/Models/LSTM/05_06_18_X-Time_N7/', 'Error_Conv+LSTM_N7_03')

        #print predicted verification values

        mean_predicted_heat=0
        for sequence_num in range(0, int(math.floor(number_of_sequences_validation/2))):
            mean_predicted_heat=heat_predicted_validation[sequence_num]+mean_predicted_heat

        print('predicted heat= '+str(100000*mean_predicted_heat/int(math.floor(number_of_sequences_validation/2)))+ ' кВт\м2   target= ' + str(target_validation[0].data[0])+' кВт/м2')

        mean_predicted_heat = 0
        for sequence_num in range(int(math.floor(number_of_sequences_validation / 2)),number_of_sequences_validation ):
            mean_predicted_heat = heat_predicted_validation[sequence_num] + mean_predicted_heat

        print('predicted heat= ' + str(100000 * mean_predicted_heat / int(
            math.floor(number_of_sequences_validation / 2))) + ' кВт\м2   target= ' + str(
            target_validation[int(math.floor(number_of_sequences_validation / 2))].data[0]) + ' кВт/м2')


        u_ii = w_ii.clone()
        u_if = w_if.clone()
        u_ic = w_ic.clone()
        u_io = w_io.clone()
        u_hi = w_hi.clone()
        u_hf = w_hf.clone()
        u_hc = w_hc.clone()
        u_ho = w_ho.clone()

        # ... after training, save your model
        torch.save([rnn,conv_layer_1,conv_layer_2,conv_layer_3,conv_layer_4,conv_layer_5], '№7_model_03.pt')



    #show first layer weigts
    weignt_1conv, bias_1conv = conv_layer_1.parameters()
    visualize.show_weights(weignt_1conv.data)

    plt.clf()
    plt.axes([0.3, 0.3, 0.5, 0.5])
    plt.title('Prediction (heat),max pool, 300 times 2 layer LSTM,Validation, epoch' + str(epoch + 1))
    plt.plot(100000*heat_predicted, 'k:', label='1')
    plt.xlabel('Heat load')
    plt.ylabel('error')
    plt.legend()
    sample_file_name = 'Prediction_LSTM_arranged_by_load' + str(steps_to_print) + '_steps_epoch_' + str(epoch) + '.png'
    #plt.savefig(results_dir + sample_file_name)
    plt.show()


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
    plt.title('iteration error by index, learning epoch' + str(epoch + 1))
    plt.plot(error, 'k:', label='1')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    plt.clf()
    plt.axes([0.3, 0.3, 0.5, 0.5])
    plt.title('iteration error heat loads, learning epoch' + str(epoch + 1))
    plt.plot(error_by_heat, 'k:', label='1')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    # here i calculate average value of prediction above middle value
    batch1 = heat_predicted[0:int(0.5 * number_of_sequences)]
    batch2 = heat_predicted[int(0.5 * number_of_sequences):number_of_sequences]
    average_predicted_load1 = numpy.mean(batch1)
    average_predicted_load2 = numpy.mean(batch2)
    predicted_load1 = numpy.mean(batch1[numpy.where(batch1 > average_predicted_load1)])
    predicted_load2 = numpy.mean(batch2[numpy.where(batch2 > average_predicted_load2)])


    print('target=' + str(100000 * target[0]) + '  predicted= ' + str(100000 * predicted_load1))
    print('target=' + str(100000 * target[-1]) + '  predicted= ' + str(100000 * predicted_load2))

