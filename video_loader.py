import cv2


def load_avi(filename):
    cap = cv2.VideoCapture(filename)

    if cap.isOpened():
        success, frame = cap.read()
    else:
        success = False
    while success:
        success, frame = cap.read()
        yield frame
        cv2.waitKey(1)
    cap.release()


if __name__ == "__main__":
    frames = load_avi('/Users/sergey/Downloads/2_ml_sq.avi')
    for f in frames:
        if f is not None:
            print(f'{len(f)}x{len(f[0])}')
