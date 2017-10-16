import numpy as np
import matplotlib.pyplot as plt
from frames_dataset import FramesDataset


def show_frame(frame, heat_transfer):
    plt.imshow(frame)


if __name__ == "__main__":
    face_dataset = FramesDataset('./train/annotations.csv', './train')

    fig = plt.figure()

    num_samples = 11
    samples_indexes = np.random.randint(len(face_dataset), size=num_samples)
    for i, index in enumerate(samples_indexes):
        sample = face_dataset[index]

        print(i, sample['frame'].shape, sample['heat_transfer'].shape)

        ax = plt.subplot(num_samples//3+1, 3, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(index))
        ax.axis('off')
        show_frame(**sample)

        if i == num_samples-1:
            plt.show()
            break
