import cv2
from os import path
from glob import glob

VIDEO_FILENAME_TEMPLATE = '*.avi'


def load_video(filename, center_crop=(256, 256)):
    """
    Возвращает все кадры ч/б видео с одной компонентой - градация серого (0-255)
    :param filename: путь к файлу
    :param center_crop: параметры кропа - вырезание центра видео
    """
    (height, width) = center_crop

    cap = cv2.VideoCapture(filename)

    success = cap.isOpened()
    while success:
        success, frame = cap.read()
        if frame is not None:
            y = max((len(frame) - height) // 2, 0)
            x = max((len(frame[0]) - width) // 2, 0)
            h = min(y + height, len(frame))
            w = min(x + width, len(frame[0]))
            yield frame[y:h, x:w, 0]/256

    cap.release()


def load_folder(dir, center_crop=(256, 256)):
    """
    Возвращает dict со всеми кадрами всех видеофайлов в директории
    :param dir: путь к директории
    :param center_crop: параметры кропа - вырезание центра видео
    """
    if not path.isdir(dir):
        return {}

    filepath = path.join(dir, VIDEO_FILENAME_TEMPLATE)
    res = {}
    for f in glob(filepath):
        res[f] = load_video(f, center_crop)
    return res


if __name__ == "__main__":
    videos = load_folder('./train')
    for name in videos:
        for frame in videos[name]:
            print(f"{len(frame)}x{len(frame[0])}")
            print(frame)
