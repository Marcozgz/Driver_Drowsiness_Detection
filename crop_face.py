"""
Crop the input images into faces.
"""
from interface import detect_face
from skimage import transform as trans
from tqdm import tqdm

import cv2
import glob
import numpy as np


def preprocess_landmark(img, landmark, img_size):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    # 计算到中心点的距离
    src[:, 0] -= 96 / 2
    src[:, 1] -= 112 / 2

    if img_size == [256, 256]:
        src[:, 0] *= 1.8
        src[:, 1] *= 1.8
        src[:, 0] += 256 / 2
        src[:, 1] += 256 / 2

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (img_size[1], img_size[0]), borderValue=0.0, flags=1)
    return warped


def warp_affine_face(image, bbox):
    _landmark = [[bbox[5], bbox[6]], [bbox[7], bbox[8]], [bbox[9], bbox[10]], [bbox[11], bbox[12]],
                 [bbox[13], bbox[14]]]
    landmark = np.array(_landmark, dtype=np.float32)
    img_size = [256, 256]
    croped = preprocess_landmark(image, landmark, img_size)
    return croped


def crop_face(path):
    imgs_path = glob.glob(path)
    for i in tqdm(range(len(imgs_path))):
        img = cv2.imread(imgs_path[i])
        h, w, _ = img.shape
        detect = detect_face("./models_file/facedetect_model.pth", h, w)
        fd_res = detect.detect(img)
        if len(fd_res) > 0:
            fd_res = np.squeeze(fd_res[0])
        else:
            continue
        cropped_img = warp_affine_face(img, fd_res)
        cv2.imwrite("D:\\dataset_new\\train_data\\no_yawn\\{}.png".format(i), cropped_img)


if __name__ == '__main__':
    data_path = "D:\\dataset_new\\train\\no_yawn\\*.jpg"
    crop_face(data_path)
