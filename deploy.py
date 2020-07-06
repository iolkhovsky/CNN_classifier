import cv2
import torch
import argparse
import numpy as np

from cnn_classifier import MnistClassifier
from utils import make_tensor_for_net

DEFAULT_MODEL = "pretrained_models/CnnClassifierMNIST_completed_2020_07_06_18_45_01.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model's checkpoint to load", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cam", help="Id of web camera being used", type=int, default=0)
    parser.add_argument("--device", help="Device being used: cuda/cpu", type=str, default="cpu")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    net = MnistClassifier()
    net.load_state_dict(torch.load(args.model))
    net.eval()
    net = net.to(args.device)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            proc_frame = np.bitwise_not(frame)
            img_tensor = make_tensor_for_net(proc_frame.copy()).to(args.device)
            prediction = net.predict(img_tensor).numpy()[0]
            max_prob_id = np.argmax(prediction)
            print("\nPrediction: ", max_prob_id, "Prob: ", prediction[max_prob_id])
            print("Distribution: ", str(prediction))

            cv2.imshow("Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
