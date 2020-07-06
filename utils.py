import datetime
import cv2
import torch
import numpy as np

from mnist_dataset import normalize_img_cyx


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def array_yxc2cyx(arr):
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 0, 1)
    return arr


def array_cyx2yxc(arr):
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr


def add_pred_marks(img, probs):
    img = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    color = (0, 255, 0)
    thickness = 1
    msg = "Prediction: " + str(np.argmax(probs)) + " prob: " + str(probs[np.argmax(probs)])
    img = cv2.putText(img, msg, (10, 10), font,
                        fontScale, color, thickness, cv2.LINE_AA)
    msg = "Distr: " + str(probs)
    img = cv2.putText(img, msg, (10, 20), font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return img


def make_tensor_for_net(cv_img, in_img_sz=(28, 28)):
    image = cv2.resize(cv_img, in_img_sz)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image = normalize_img_cyx(image)
    in_tensor = torch.from_numpy(image).reshape(1, 1, in_img_sz[0], in_img_sz[1])
    return in_tensor
