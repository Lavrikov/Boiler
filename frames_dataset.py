from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import picture_transformation
import numpy as np


class FramesDataset(Dataset):
    """
    Датасет отдельных кадров видео
    """
    def __init__(self, annotations_csv_file, video_dir, transform=None):
        """
        :param annotations_csv_file: путь к файлу с аннотациями к видеофайлам
        :param video_dir: путь к папке с видеофайлами
        :param transform: опциональное преобразование над файлами
        """
        self.annotations_df = pd.read_csv(annotations_csv_file, usecols=['filename', 'heat_transfer','date'])
        self.root_dir = video_dir
        self.transform = transform

        self.videos_cache = dict()

        self.annotations_df['end_frame_index'] = \
            self.annotations_df['filename']\
                .apply(lambda f: self.get_frames_count(self.root_dir, f))\
                .cumsum()

        self.annotations_df['start_frame_index'] = \
            self.annotations_df['end_frame_index']\
                .shift(1)\
                .fillna(0)\
                .astype(int)

        print(self.annotations_df.head())

    @staticmethod
    def get_frames_count(root_dir, filename):
        filepath = os.path.join(root_dir, filename)
        print(filepath)
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return 0

    def __len__(self):
        return self.annotations_df['end_frame_index'].max()

    def __getitem__(self, idx):
        videofile = self.annotations_df[(self.annotations_df['start_frame_index'] <= idx) & (idx < self.annotations_df['end_frame_index'])]
        filename = videofile['filename'].values[0]
        heat_transfer = videofile['heat_transfer'].values[0]
        frame_index = idx - videofile['start_frame_index'].values[0]

        if filename not in self.videos_cache:

            #here i add instuction to del all cached video exept new one
            #if len(self.videos_cache.keys()) > 0: del self.videos_cache[list(self.videos_cache.keys())[0]]

            self.videos_cache[filename] = dict()

            filepath = os.path.join(self.root_dir, filename)
            cap = cv2.VideoCapture(filepath)

            success = cap.isOpened()
            i = 0
            while success:
                success, frame = cap.read()
                if frame is not None:

                    if videofile['date'].values[0]=="08_06_18":
                        self.videos_cache[filename][i] = picture_transformation.resizeX(frame[-22:-8, :, 0])#-11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing
                    else:
                        #self.videos_cache[filename][i] = picture_transformation.resizeX(frame[-17:-3, :, 0])  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing
                        self.videos_cache[filename][i] = picture_transformation.resizeX(frame[:, :, 0])  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                    i += 1
            cap.release()

        frame = self.videos_cache[filename][frame_index]

        sample = {'frame': frame, 'heat_transfer': heat_transfer}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FramesDataset_Mono(Dataset):
    """
    Датасет отдельных кадров видео, кдры вытянуты в одномерный массив для удобства работы с RNN
    """
    def __init__(self, annotations_csv_file, video_dir, transform=None):
        """
        :param annotations_csv_file: путь к файлу с аннотациями к видеофайлам
        :param video_dir: путь к папке с видеофайлами
        :param transform: опциональное преобразование над файлами
        """
        self.annotations_df = pd.read_csv(annotations_csv_file, usecols=['filename', 'heat_transfer','date'])
        self.root_dir = video_dir
        self.transform = transform

        self.videos_cache = dict()

        self.annotations_df['end_frame_index'] = \
            self.annotations_df['filename']\
                .apply(lambda f: self.get_frames_count(self.root_dir, f))\
                .cumsum()

        self.annotations_df['start_frame_index'] = \
            self.annotations_df['end_frame_index']\
                .shift(1)\
                .fillna(0)\
                .astype(int)

        print(self.annotations_df.head())

    @staticmethod
    def get_frames_count(root_dir, filename):
        filepath = os.path.join(root_dir, filename)
        print(filepath)
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return 0

    def __len__(self):
        return self.annotations_df['end_frame_index'].max()

    def __getitem__(self, idx):
        videofile = self.annotations_df[(self.annotations_df['start_frame_index'] <= idx) & (idx < self.annotations_df['end_frame_index'])]
        filename = videofile['filename'].values[0]
        heat_transfer = videofile['heat_transfer'].values[0]
        frame_index = idx - videofile['start_frame_index'].values[0]

        if filename not in self.videos_cache:

            #here i add instuction to del all cached video exept new one
            #if len(self.videos_cache.keys()) > 0: del self.videos_cache[list(self.videos_cache.keys())[0]]

            self.videos_cache[filename] = dict()

            filepath = os.path.join(self.root_dir, filename)
            cap = cv2.VideoCapture(filepath)

            success = cap.isOpened()
            i = 0
            while success:
                success, frame = cap.read()

                if frame is not None:

                    #self.videos_cache[filename][i] = picture_transformation.resizeX(frame[-17:-3, :, 0])  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                    cach=picture_transformation.resizeX(frame[:, :, 0])
                    self.videos_cache[filename][i] = np.resize(cach,cach.shape[0]*cach.shape[1])  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                    i += 1
            cap.release()

        frame = self.videos_cache[filename][frame_index]

        sample = {'frame': frame, 'heat_transfer': heat_transfer}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FramesDataset_Conv3D(Dataset):
    """
    Датасет отдельных кадров видео
    """
    def __init__(self, annotations_csv_file, video_dir, transform=None):
        """
        :param annotations_csv_file: путь к файлу с аннотациями к видеофайлам
        :param video_dir: путь к папке с видеофайлами
        :param transform: опциональное преобразование над файлами
        """
        self.annotations_df = pd.read_csv(annotations_csv_file, usecols=['filename', 'heat_transfer','date'])
        self.root_dir = video_dir
        self.transform = transform

        self.videos_cache = dict()

        self.annotations_df['end_frame_index'] = \
            self.annotations_df['filename']\
                .apply(lambda f: self.get_frames_count(self.root_dir, f))\
                .cumsum()

        self.annotations_df['start_frame_index'] = \
            self.annotations_df['end_frame_index']\
                .shift(1)\
                .fillna(0)\
                .astype(int)

        print(self.annotations_df.head())

    @staticmethod
    def get_frames_count(root_dir, filename):
        filepath = os.path.join(root_dir, filename)
        print(filepath)
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/3)
        else:
            return 0

    def __len__(self):
        return self.annotations_df['end_frame_index'].max()

    def __getitem__(self, idx):
        videofile = self.annotations_df[(self.annotations_df['start_frame_index'] <= idx) & (idx < self.annotations_df['end_frame_index'])]
        filename = videofile['filename'].values[0]
        heat_transfer = videofile['heat_transfer'].values[0]
        frame_index = idx - videofile['start_frame_index'].values[0]

        if filename not in self.videos_cache:

            #here i add instuction to del all cached video exept new one
            #if len(self.videos_cache.keys()) > 0: del self.videos_cache[list(self.videos_cache.keys())[0]]

            self.videos_cache[filename] = dict()

            filepath = os.path.join(self.root_dir, filename)
            cap = cv2.VideoCapture(filepath)

            success = cap.isOpened()
            i = 0

            while success:

                success, frame = cap.read()
                if frame is not None:

                    self.videos_cache[filename][i] = picture_transformation.resizeX(frame[:, :, 0])  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                success, frame = cap.read()
                if frame is not None:

                    self.videos_cache[filename][i] = np.dstack ((self.videos_cache[filename][i],picture_transformation.resizeX(frame[:, :, 0])))  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                success, frame = cap.read()
                if frame is not None:

                    self.videos_cache[filename][i] = np.dstack ((self.videos_cache[filename][i],picture_transformation.resizeX(frame[:, :, 0])))  # -11:-1 desrease the vertical size of the image by cut it, decreze horizontal size by resizing

                i += 1
            cap.release()

        frame = self.videos_cache[filename][frame_index]

        sample = {'frame': frame, 'heat_transfer': heat_transfer}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    frames_dataset = FramesDataset('file:///media/aleksandr/Files/@Machine/Github/Boiler/train/annotations.csv', 'file:///media/aleksandr/Files/@Machine/Github/Boiler/train')
    print(frames_dataset.__getitem__(360000))



