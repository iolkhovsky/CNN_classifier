import cv2
import torch
import numpy as np
import argparse

from cnn_classifier import MnistClassifier
from utils import make_tensor_for_net


DEFAULT_MODEL = "pretrained_models/CnnClassifierMNIST_completed_2020_07_06_14_56_16.pt"


class OcrProcessor:

    def __init__(self, path_to_model=DEFAULT_MODEL):
        self.threshold = 127
        self.cnn = MnistClassifier()
        self.cnn.load_state_dict(torch.load(path_to_model))
        self.cnn.eval()
        return

    def __call__(self, *args, **kwargs):
        img = args[0]
        res = self.process(img.copy())
        for bbox, label, prob in res:
            x, y, w, h = bbox
            color = (0, 0, 255)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
            msg1 = "Symbol: " + str(label)
            msg2 = "(" + str("{:.3f}".format(prob)) + ")"
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, msg1, (int(x), int(y - 30)), font,
                              fontScale=0.7, color=color, thickness=1, lineType=cv2.LINE_AA)
            img = cv2.putText(img, msg2, (int(x), int(y - 10)), font,
                              fontScale=0.7, color=color, thickness=1, lineType=cv2.LINE_AA)
        return img

    def process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        _, binary = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY_INV)
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        out = []
        with torch.no_grad():
            for idx in np.arange(1, cnt):
                region_stats = stats[idx]
                w, h = region_stats[2], region_stats[3]
                x, y = centroids[idx][0], centroids[idx][1]

                if max(w, h) < 28:
                    continue

                max_sz = 1.2 * max(w, h)
                proc_frame = img[int(y - 0.5 * max_sz): int(y + 0.5 * max_sz),
                             int(x - 0.5 * max_sz): int(x + 0.5 * max_sz)].copy()
                proc_frame = np.bitwise_not(proc_frame)


                img_tensor = make_tensor_for_net(proc_frame.copy())
                prediction = self.cnn.predict(img_tensor).numpy()[0]
                label = np.argmax(prediction)
                prob = prediction[label]
                out.append(((x - 0.5 * w, y - 0.5 * h, w, h), label, prob))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model's checkpoint to load", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--image", help="Path to image being processed", type=str, default="attachments/ocr_test.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    processor = OcrProcessor(path_to_model=args.model)
    result = processor(img)
    cv2.imwrite("attachments/ocr_result.jpg", result)
    return


if __name__ == "__main__":
    main()
