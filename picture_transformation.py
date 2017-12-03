import cv2


def boundaries_detect_laplacian(sample):
    # select the boundaries of objects in the frame
    LaplacianFilterSample = cv2.Laplacian(sample['frame'], 3)

    # zero all pixels that are not boundaries of objects
    ret, BinareFilterSample = cv2.threshold(LaplacianFilterSample, 127, 255, cv2.THRESH_BINARY)

    return BinareFilterSample