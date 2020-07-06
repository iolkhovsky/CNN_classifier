import torch
from torch.nn import CrossEntropyLoss
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from cnn_classifier import MnistClassifier
from metrics import accuracy
from mnist_dataset import denormalize_image, get_test_dataloader
from utils import add_pred_marks, array_yxc2cyx


DEFAULT_MODEL = "pretrained_models/CnnClassifierMNIST_completed_2020_07_06_14_56_16.pt"


def test(model, device, test_loader, batches=None):
    model.eval()
    test_loss = 0
    res = {}
    correct = 0
    loss_function = CrossEntropyLoss(reduction='sum')
    batch_size = test_loader.batch_size

    y_target = []
    y_predicted = []

    with torch.no_grad():
        val_imgs = []
        for idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = model.predict(input)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            acc = accuracy(pred, target, norm=False)
            correct += acc

            y_target += list(target)
            y_predicted += list(pred.flatten())

            val_imgs = []
            for i, img in enumerate(input):
                img = denormalize_image(img[0])
                img = add_pred_marks(img, output[i].numpy())
                img = array_yxc2cyx(img)
                val_imgs.append(torch.from_numpy(img))

            if batches is not None:
                if idx + 1 >= batches:
                    break

        res["cm"] = confusion_matrix(y_target, y_predicted)

        if batches is not None:
            test_loss = test_loss / (batches * batch_size)
            res["accuracy"] = correct / (batches * batch_size)
        else:
            test_loss = test_loss / (len(test_loader) * batch_size)
            res["accuracy"] = correct / (len(test_loader) * batch_size)
        res["images"] = val_imgs

    return test_loss, res


def test_with_bar(model, device, test_loader, batches=None):
    model.eval()
    test_loss = 0
    res = {}
    correct = 0
    loss_function = CrossEntropyLoss(reduction='sum')
    batch_size = test_loader.batch_size

    y_target = []
    y_predicted = []

    with torch.no_grad():

        val_imgs = []
        total_cnt = len(test_loader)
        if batches:
            total_cnt = min(total_cnt, batches)

        with tqdm(total=total_cnt * test_loader.batch_size,
                  desc="Model's validation",
                  unit='image') as pbar:

            for idx, (input, target) in enumerate(test_loader):
                input, target = input.to(device), target.to(device)
                output = model.predict(input)
                test_loss += loss_function(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                acc = accuracy(pred, target, norm=False)
                correct += acc

                y_target += list(target)
                y_predicted += list(pred.flatten())

                val_imgs = []
                for i, img in enumerate(input):
                    img = denormalize_image(img[0])
                    img = add_pred_marks(img, output[i].numpy())
                    img = array_yxc2cyx(img)
                    val_imgs.append(torch.from_numpy(img))

                pbar.update(test_loader.batch_size)

                if batches is not None:
                    if idx + 1 >= batches:
                        break

        res["cm"] = confusion_matrix(y_target, y_predicted)

        if batches is not None:
            test_loss = test_loss / (batches * batch_size)
            res["accuracy"] = correct / (batches * batch_size)
        else:
            test_loss = test_loss / (len(test_loader) * batch_size)
            res["accuracy"] = correct / (len(test_loader) * batch_size)
        res["images"] = val_imgs

    return test_loss, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model's checkpoint to load", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", help="Device being used: cuda/cpu", type=str, default="cpu")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=32)
    args = parser.parse_args()

    model = MnistClassifier()
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    test_dloader = get_test_dataloader(path="dataset", batch_size=args.batch_size)
    val_loss, result = test_with_bar(model, args.device, test_dloader, batches=None)
    print("Loss: ", val_loss, " Acc: ", result["accuracy"])
    print("Confusion matrix: ")
    print(result["cm"])
    return


if __name__ == "__main__":
    main()
