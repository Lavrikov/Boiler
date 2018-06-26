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


def test_dataset(face_dataset,from_frame,video_length):
    # visualize some data
    sample = face_dataset[1]
    print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    for j in range(0, 1):
        for i in range(1, 11):
            # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
            SummResult = face_dataset[from_frame + video_length* i + video_length * 10 * j ]['frame']
            # here i show results
            ax = plt.subplot(11 // 3 + 1, 3, i)  # coordinates
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i + 10 * j))
            ax.axis('off')
            # show the statistic matrix
            plt.imshow(SummResult, 'gray')

        plt.show()


def regularization_penalty(hn,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y):


    penalty=0.0
    #penalties 1 layer
    from_hn_features=0
    for i in range(0, reg_layer2_y):
        for j in range(0, reg_layer2_x):

            k = from_hn_features + i * reg_layer2_y + j

            k_1 = from_hn_features + (i+1) * reg_layer2_y + j

            k_2 = reg_layer1_x * reg_layer1_y + i * reg_layer2_y + j

            w_x = hn[0,0,k] + hn[0,0,k+1]

            w_y = hn[0,0,k_1] + hn[0,0,k_1+1]

            w_z = hn[0,0,k_2]

            penalty = penalty + torch.abs(w_x + w_y + w_z)


    # penalties 2 layer
    penalty_2 = 0.0
    from_hn_layer2 = from_hn_features + reg_layer2_x * reg_layer2_y
    for i in range(0, reg_layer2_y):
        for j in range(0, reg_layer2_x):

            k = from_hn_layer2 + i * reg_layer2_y + j

            k_1 = from_hn_layer2 + (i+1) * reg_layer2_y + j

            k_2 = from_hn_features + i * reg_layer2_y + j

            F = 0.5 * (hn[0,0,k] + hn[0,0,k+1]) + 0.5 * (hn[0,0,k_1] + hn[0,0,k_1+1])

            w_z = hn[0,0,k_2]

            penalty_2 = penalty_2 + w_z * F

    penalty = penalty + torch.abs(penalty_2)


    # penalties 3 layer
    penalty_3 = 0.0

    from_hn_layer2 = from_hn_features + reg_layer2_x * reg_layer2_y

    for i in range(0, reg_layer3_y):
        for j in range(0, reg_layer3_x):
            k = from_hn_layer2 + i * reg_layer3_y + j

            F = torch.abs(hn[0,0,k] - torch.abs(hn[0,0,k]))

            penalty_3 = penalty_3 + F

    penalty = penalty + penalty_3

    penalty = penalty * penalty

    return penalty


def target_generator( face_dataset, number_of_sequences, number_of_farme_per_batch, number_of_sequences_validation):

    target, target_validation = torch.cuda.FloatTensor(number_of_sequences), torch.cuda.FloatTensor(number_of_sequences_validation)

    print('target generation (heat load)')
    for sequence_num in range(0, number_of_sequences):
        sample_num = first_sample_lstm + sequence_num * number_of_farme_per_batch
        print(sample_num)
        target[sequence_num] = float(face_dataset[sample_num]['heat_transfer']) / 100000

        # put all video to one tensor (GPU or not)
        # for i in range(0, number_of_farme_per_batch):
        #    input_captured[sequence_num,i, 0] = torch.from_numpy(face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'])

        # print(sample_num)

    print('target validation generation (heat load)')
    for sequence_num in range(0, number_of_sequences_validation):
        sample_num = first_sample_lstm_validation + sequence_num * number_of_farme_per_batch
        print(sample_num)
        target_validation[sequence_num] = float(face_dataset[sample_num]['heat_transfer']) / 100000


    return Variable(target), Variable(target_validation)


def forward(input):

    output = torch.nn.functional.normalize(input)
    output = model_convolutional_3D(output)
    #resize
    (B, F, D, H, W) = output.data.size()
    output = output.view(-1, B * F * D * H * W)
    output=fully_connected_layer_1(output)

    return output

def test_convolutional_part( face_dataset ,number_of_farme_per_batch ,first_sample_lstm):

    input = Variable(
        torch.cuda.FloatTensor(1, 1, number_of_farme_per_batch, face_dataset[first_sample_lstm]['frame'].shape[0],
                               face_dataset[first_sample_lstm]['frame'].shape[1]).zero_())

    print('size of one dimension image' + str(input.data.shape[2]))
    for i in range(0, number_of_farme_per_batch):
        input.data[0, 0, i] = torch.from_numpy(
            face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'])

    print('input')
    print(input.shape)

    print ('test norm_1')
    output=torch.nn.functional.normalize(input)
    print(output.shape)

    print ('test conv_3D')
    output= model_convolutional_3D(output)
    print (output.shape)

    return output,input



if __name__ == "__main__":

    #device = torch.device('cpu')
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')

    print('Cuda available?  '+ str(torch.cuda.is_available())+ ', videocard  '+ str(torch.cuda.device_count()))

    video_length = 12000
    number_of_samples_lstm, first_sample_lstm = 86 * video_length, 0 * video_length #86
    number_of_samples_lstm_validation, first_sample_lstm_validation = 19 * video_length, 87 * video_length #19


    #here i load the video dataset like a group of a pictures and view some pictures
    basePath=os.path.dirname(os.path.abspath(__file__))
    face_dataset = FramesDataset(basePath+'/train/annotations.csv',basePath+ '/train')
    test_dataset(face_dataset,first_sample_lstm,video_length)


    #below I init model parts
    zero_load_repeat,number_of_farme_per_batch,F_1= 5, 300, 20
    number_of_sequences=int(math.floor(number_of_samples_lstm/number_of_farme_per_batch))
    number_of_sequences_validation=int(math.floor(number_of_samples_lstm_validation/number_of_farme_per_batch))
    error, error_by_heat, heat_predicted = torch.cuda.FloatTensor(number_of_sequences),torch.cuda.FloatTensor(number_of_sequences), torch.cuda.FloatTensor(number_of_sequences)
    error_validation, error_by_heat_validation, heat_predicted_validation =torch.cuda.FloatTensor(number_of_sequences_validation),torch.cuda.FloatTensor(number_of_sequences_validation),torch.cuda.FloatTensor(number_of_sequences_validation)

    sequence_num=0

    # The 3D Convolutional

    model_convolutional_3D = torch.nn.Sequential(
        torch.nn.Conv3d(1, F_1, 3),
        torch.nn.MaxPool3d((2, 1, 2)),
        torch.nn.Conv3d(F_1, F_1*2, 3),
        torch.nn.MaxPool3d((2, 1, 2)),
        torch.nn.Conv3d(F_1*2, F_1*4, 3),
        torch.nn.MaxPool3d((2, 1, 2)),
        torch.nn.Conv3d(F_1*4, F_1*8, 3),
        torch.nn.MaxPool3d((2, 1, 2)),
        torch.nn.Conv3d(F_1*8, F_1*16, 3),
        torch.nn.MaxPool3d((2, 2, 2)),
    ).cuda()

    output, input = test_convolutional_part(face_dataset, number_of_farme_per_batch, first_sample_lstm)

    print('resize')
    (B, F, D, H, W) = output.data.size()
    output = output.view(-1, B * F * D * H * W)
    print(output)

    fully_connected_layer_1 = torch.nn.Linear(output.shape[1], 1).cuda()

    output=fully_connected_layer_1(output)

    output = forward(input)

    print(output)

    test_dataset(face_dataset, first_sample_lstm, video_length)

    optimizerLSTM=torch.optim.Adadelta([
                                        {'params': model_convolutional_3D.parameters()},
                                        {'params': fully_connected_layer_1.parameters()}
                                        ], lr=0.001)


    # load pretrained model if it is required
    #[rnn, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5] = torch.load('№7_model_01.pt')
    #rnn.flatten_parameters()


    #Cycle parameters
    target, target_validation = target_generator(face_dataset, number_of_sequences, number_of_farme_per_batch, number_of_sequences_validation)
    epoch_number, steps_to_print=90, number_of_sequences-1
    train_vs_epoch,validation_vs_epoch=torch.cuda.FloatTensor(epoch_number).zero_(), torch.cuda.FloatTensor(epoch_number).zero_()


    print('train started Convolution')
    # repead cycle by all samples
    for epoch in range(0,epoch_number):
        print('learning epoch'+str(epoch+1))

        samples_indexes = [i for i in range(0, number_of_sequences)]  # A list contains all shuffled requires numbers
        shuffle(samples_indexes)


        for index, sequence_num in enumerate(samples_indexes):

            #print(str(index)+' from '+ str(number_of_sequences))
            #print('createinput')
            for i in range(0, number_of_farme_per_batch):
                input.data[0, 0, i] = torch.from_numpy(
                    face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'])
            #print('conv_started')
            output = forward(input)
            #print('calc loss')
            loss = ((target[sequence_num]) - output) ** 2# + regularization_penalty( hn ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y)

            print (str(index)+' loss' + str(loss.data[0,0])) #+ ' regularization penalty ' + str(regularization_penalty( hn ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y).data[0]))

            error[index] = loss.data[0,0]

            error_by_heat[sequence_num] = ((target[sequence_num]) - output).data[0,0]

            heat_predicted[sequence_num]=torch.max((output)).data[0]
            #print('backpropagation calc')
            loss.backward()

            #print('renew weight')
            optimizerLSTM.step()

            optimizerLSTM.zero_grad()

            if index==100*int(index/100): print(index)
            #print('cycle ended')
            #here i repeat passing through zero load elements, to increase their weight at the all data

            if target.data[sequence_num]==0 :

                for zero_repeat in range(0,zero_load_repeat):

                    output = forward(input)

                    loss = ((target[sequence_num]) - output) ** 2 #+  regularization_penalty( hn ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y)

                    error[index] = loss.data[0, 0]

                    error_by_heat[sequence_num] = ((target[sequence_num]) - output).data[0, 0]

                    heat_predicted[sequence_num] = torch.max((output)).data[0]

                    loss.backward()

                    optimizerLSTM.step()

                    optimizerLSTM.zero_grad()


        print('validation epoch' + str(epoch + 1))

        for sequence_num in range(0,number_of_sequences_validation):

            #print(str(index)+' from '+ str(number_of_sequences))

            for i in range(0, number_of_farme_per_batch):
                input.data[0, 0, i] = torch.from_numpy(
                    face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'])

            output = forward(input)

            loss = ((target_validation[sequence_num]) - output) ** 2#  + regularization_penalty( hn ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y)

            error_validation[sequence_num] = loss.data[0, 0]

            error_by_heat_validation[sequence_num] = ((target_validation[sequence_num]) - output).data[0, 0]

            heat_predicted_validation[sequence_num] = torch.max((output)).data[0]

            if sequence_num == 100 * int(sequence_num / 100): print(sequence_num)

        visualize.save_some_epoch_data(index, number_of_sequences-1, epoch, basePath, '/Models/LSTM/24_06_18_X-Time_N12/', 'Error_Conv+LSTM_N12_07', error_validation.cpu().numpy(), error_by_heat_validation.cpu().numpy(), 'verification','Conv 6 +fully_conn, *5 zero load,')


        #here i create figure with the history of training and validation

        train_vs_epoch[epoch] = torch.mean(torch.abs(error_by_heat))

        validation_vs_epoch[epoch]=torch.mean(torch.abs(error_by_heat_validation))

        visualize.save_train_validation_picture(train_vs_epoch.cpu().numpy()[0:epoch+1],validation_vs_epoch.cpu().numpy()[0:epoch+1], basePath, '/Models/LSTM/24_06_18_X-Time_N12/', 'Error_Conv+LSTM_N12_07')



        # print predicted verification values

        mean_predicted_heat = 0
        heat_sec_num = int(math.floor(video_length / number_of_farme_per_batch))
        for heat_load in range(0, int(math.floor(number_of_samples_lstm_validation / video_length))):

            for sequence_num in range(heat_sec_num * heat_load, heat_sec_num * (heat_load + 1)):
                mean_predicted_heat = heat_predicted_validation[      sequence_num] + mean_predicted_heat

            print('predicted heat= ' + str(100000 * mean_predicted_heat / int(
                math.floor(number_of_sequences_validation / 2))) + ' Вт\м2   target= ' + str(
                100000*target_validation[heat_sec_num * heat_load].data[0]) + ' Вт/м2')



        # ... after training, save your model
        torch.save([model_convolutional_3D, fully_connected_layer_1], '№12_model_07.pt')




