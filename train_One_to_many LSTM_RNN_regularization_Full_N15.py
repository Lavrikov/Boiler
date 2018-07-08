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
from matplotlib import animation




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


def forward(input,number_of_farme_per_batch):

    reg = fully_connected_layer1(input)
    D = reg.data.size()
    B= int(math.floor(D[0]/number_of_farme_per_batch))
    reg = reg.view(number_of_farme_per_batch, 1, B)

    output, (hn, cn) = LSTM(reg)


    return reg, output


def init():
    plot.set_data(data[0])
    return [plot]


def update(j):
    plot.set_data(data[j])
    return [plot]




if __name__ == "__main__":

    #device = torch.device('cpu')
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')

    print('Cuda available?  '+ str(torch.cuda.is_available())+ ', videocard  '+ str(torch.cuda.device_count()))

    video_length = 12000
    number_of_samples_lstm, first_sample_lstm = 19 * video_length, 33 * video_length #19
    number_of_samples_lstm_validation, first_sample_lstm_validation = 2 * video_length, 87 * video_length #19


    #here i load the video dataset like a group of a pictures and view some pictures
    basePath=os.path.dirname(os.path.abspath(__file__))
    face_dataset = FramesDataset(basePath+'/train/annotations.csv',basePath+ '/train')
    #test_dataset(face_dataset,first_sample_lstm,video_length)


    #below I init model parts
    zero_load_repeat,number_of_farme_per_batch=0, 300,
    number_of_sequences=int(math.floor(number_of_samples_lstm/number_of_farme_per_batch))
    number_of_sequences_validation=int(math.floor(number_of_samples_lstm_validation/number_of_farme_per_batch))
    error, error_by_heat, heat_predicted = torch.cuda.FloatTensor(number_of_sequences),torch.cuda.FloatTensor(number_of_sequences), torch.cuda.FloatTensor(number_of_sequences)
    error_validation, error_by_heat_validation, heat_predicted_validation =torch.cuda.FloatTensor(number_of_sequences_validation),torch.cuda.FloatTensor(number_of_sequences_validation),torch.cuda.FloatTensor(number_of_sequences_validation)

    input = Variable(torch.cuda.FloatTensor(number_of_farme_per_batch, 1, face_dataset[first_sample_lstm]['frame'].shape[0]*face_dataset[first_sample_lstm]['frame'].shape[1]).zero_())

    sequence_num=0
    print('size of one dimension image'+str(input.data.shape[2]))
    for i in range(0, number_of_farme_per_batch):

        input.data[i, 0] = torch.from_numpy(np.resize(face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'],input.data.shape[2]))

    print('input')
    print(input)

    #regularisation parameters of LSTM hidden Layer
    reg_layer1_x, reg_layer1_y = 40, 12
    reg_layer2_x, reg_layer2_y, reg_layer3_x,reg_layer3_y  = reg_layer1_x-1, reg_layer1_y-1, reg_layer1_x, reg_layer1_y


    # The LSTM model part
    hidden_layer, hidden_features= 1, reg_layer1_x * reg_layer1_y + reg_layer2_x * reg_layer2_y + reg_layer3_x * reg_layer3_y
    print('hidden_features='+str(hidden_features))

    LSTM= torch.nn.LSTM(hidden_features, input.data.shape[2], hidden_layer).cuda()
    fully_connected_layer1= torch.nn.Linear(1, hidden_features*number_of_farme_per_batch).cuda()
    loss_new=torch.nn.SmoothL1Loss()

    optimizerLSTM=torch.optim.Adadelta([
                                        {'params': LSTM.parameters()},
                                        {'params': fully_connected_layer1.parameters()}
                                        ], lr=1)


    # load pretrained model if it is required
    #[rnn, conv_layer_1, conv_layer_2, conv_layer_3, conv_layer_4, conv_layer_5] = torch.load('№7_model_01.pt')
    #rnn.flatten_parameters()


    #Cycle parameters
    target, target_validation = target_generator(face_dataset, number_of_sequences, number_of_farme_per_batch, number_of_sequences_validation)
    epoch_number, steps_to_print=90, number_of_sequences-1
    train_vs_epoch,validation_vs_epoch=torch.cuda.FloatTensor(epoch_number).zero_(), torch.cuda.FloatTensor(epoch_number).zero_()


    print('train started Convolution+LSTM')
    # repead cycle by all samples
    for epoch in range(0,epoch_number):
        print('learning epoch'+str(epoch+1))

        samples_indexes = [i for i in range(0, number_of_sequences)]  # A list contains all shuffled requires numbers
        shuffle(samples_indexes)


        for index, sequence_num in enumerate(samples_indexes):

            print(str(index)+' from '+ str(number_of_sequences))

            for i in range(0, number_of_farme_per_batch):
                input.data[i, 0] = torch.from_numpy(np.resize(face_dataset[first_sample_lstm + sequence_num * number_of_farme_per_batch + i]['frame'], input.data.shape[2]))
            #input=Variable(input_captured_pinned[sequence_num])

            input = 0.01 * (input - torch.mean(input)) / torch.max(torch.abs(input))

            reg, output = forward(target[sequence_num],number_of_farme_per_batch)


            loss = torch.sum((input - output)**2)# + regularization_penalty(reg ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y)

            print ('loss' + str(loss.data[0]) + ' regularization penalty ' + str(regularization_penalty( reg ,reg_layer1_x, reg_layer1_y,reg_layer2_x, reg_layer2_y,reg_layer3_x, reg_layer3_y).data[0]))

            error[index] = loss.data[0]

            #error_by_heat[sequence_num] = ((target[sequence_num]) - output).data[0]

            #heat_predicted[sequence_num]=torch.max((output)).data[0]

            loss.backward()

            optimizerLSTM.step()

            optimizerLSTM.zero_grad()

            if index == 10 * int(index / 10):

                # show generated video
                print('show generated video')
                data = np.empty(number_of_farme_per_batch, dtype=object)  # Almacena los datos

                # Leer todos los datos
                for k in range(number_of_farme_per_batch):
                    reg = output.data[k,0]
                    reg = reg.view(face_dataset[first_sample_lstm]['frame'].shape[0], face_dataset[first_sample_lstm]['frame'].shape[1])
                    data[k]=reg.cpu().numpy()


                fig = plt.figure()
                plot = plt.matshow(data[0], cmap='gray', fignum=0)
                plot.set_title(str(target[sequence_num].data)+' W/m2')

                anim = animation.FuncAnimation(fig, update, init_func=init, frames=number_of_farme_per_batch, interval=30, blit=True)

                plt.show()

        visualize.save_some_epoch_data(index, number_of_sequences-1, epoch, basePath, '/Models/LSTM/20_06_18_X-Time_N11/', 'Error_Conv+LSTM_N15_01', error_validation.cpu().numpy(), error_by_heat_validation.cpu().numpy(), 'verification','Conv 4 + LSTM_+fully_conn, *5 zero load,')


        #here i create figure with the history of training and validation

        train_vs_epoch[epoch] = torch.mean(torch.abs(error_by_heat))

        validation_vs_epoch[epoch]=torch.mean(torch.abs(error_by_heat_validation))

        visualize.save_train_validation_picture(train_vs_epoch.cpu().numpy()[0:epoch+1],validation_vs_epoch.cpu().numpy()[0:epoch+1], basePath, '/Models/LSTM/20_06_18_X-Time_N11/', 'Error_Conv+LSTM+Fully_con_N15_01')



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
        torch.save([LSTM, fully_connected_layer1], '№15_model_01.pt')




