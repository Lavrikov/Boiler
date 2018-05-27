import matplotlib.pyplot as plt
import torch

from frames_dataset import FramesDataset
from picture_transformation import boundaries_detect_laplacian
from train_k_neihbor import boundaries_summ_conv


def show_frame(frame, heat_transfer):
    plt.imshow(frame,'gray')


def show_loss(index, wi, wf, wc, wo, hi, hf, hc, ho, sequence_num, first_sample_lstm, time_step_speed, time_step_signature, loss, hn, error_by_heat, target):

    print(str(index) + '  ' + str(
        first_sample_lstm + sequence_num * time_step_speed * time_step_signature) + '-' + str(
        first_sample_lstm + (sequence_num + 1) * time_step_speed * time_step_signature) + ' ' + str(
        "%.4f" % torch.sum(wi).data[0]) + '  ' + str(
        "%.4f" % torch.sum(wf).data[0]) + '  ' + str(
        "%.4f" % torch.sum(wc).data[0]) + '  ' + str(
        "%.4f" % torch.sum(wo).data[0]) + '  ' + str(
        "%.4f" % torch.sum(hi).data[0]) + '  ' + str(
        "%.4f" % torch.sum(hf).data[0]) + '  ' + str(
        "%.4f" % torch.sum(hc).data[0]) + '  ' + str(
        "%.4f" % torch.sum(ho).data[0]) + '  loss=' + str(
        "%.4f" % loss.data[0]) + '  out' + str("%.4f" % torch.max(hn).data[0]) + '   error=' + str(
        error_by_heat.data[sequence_num]) + '   heat=' + str(100000 * target[sequence_num].data[0]))


def save_some_epoch_data(current_cycle_step, every_step_save, epoch, basePath, folder, file_name, error, error_by_heat, run_key, title):

    """
        :param current_cycle_step: the data are saved on the particular step, the step number is calculated inside function
        :param every_step_save: condition to calculate the step number that will be saved
        :param epoch: data that be saved
        :param basePath: path to file folder where code was run
        :param folder: folder name to the particular calculating case, need be created manually
        :param file_name: part of the file name that covers calculation conditions
        :param error: data that be saved
        :param error_by_heat: data that be saved
        :param run_key: data that be saved (train, validation,...
        :param title: name for the picture
    """

    if current_cycle_step == every_step_save * int(current_cycle_step / every_step_save):
        plt.clf()
        plt.axes([0.3, 0.3, 0.5, 0.5])
        plt.title('loss(index),' + title + ',' + str(run_key) + str(epoch + 1))
        plt.plot(error, 'k:', label='1')
        plt.xlabel('Iteration')
        plt.ylabel('loss')
        plt.legend()
        results_dir = basePath + folder
        sample_file_name = str(run_key) + file_name + str(every_step_save) + '_steps_epoch_' + str(
            epoch) + '.png'
        plt.savefig(results_dir + sample_file_name)

        plt.clf()
        plt.axes([0.3, 0.3, 0.5, 0.5])
        plt.title('error (heat),' + title + ',' + str(run_key) + str(epoch + 1))
        plt.plot(error_by_heat, 'k:', label='1')
        plt.xlabel('Heat load')
        plt.ylabel('error')
        plt.legend()
        sample_file_name = str(run_key) + file_name+ '_arranged_by_load' + str(
            every_step_save) + '_steps_epoch_' + str(epoch) + '.png'
        plt.savefig(results_dir + sample_file_name)
        print('Saved  step=' + str(current_cycle_step) + '  epoch='+str(epoch))


if __name__ == "__main__":
    #here i load the video dataset like a group of a pictures
    face_dataset = FramesDataset('file:///media/aleksandr/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/aleksandr/Files/@Machine/Github/Boiler/train')

    # here i calculate statistics of bubble boundaries appeariance at every coordinate of image with multiplication by 1000
    SummResult=boundaries_summ_conv(face_dataset,63 * 12000, 64 * 12000, 1000)

    sample=face_dataset[1]
    fig = plt.figure()
    print(1, sample['frame'].shape, sample['heat_transfer'].shape)
    ax = plt.subplot(11 // 3 + 1, 3, 1 + 1) #coordinates
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(1))
    ax.axis('off')
    print(SummResult)

    # show the statistic matrix
    sample['frame'] = SummResult.numpy()
    show_frame(**sample)
    plt.show()

