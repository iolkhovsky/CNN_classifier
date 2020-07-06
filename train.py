import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from cnn_classifier import MnistClassifier
from mnist_dataset import get_train_dataloader, get_test_dataloader
from torch.optim import Adam, SGD

from test import test
from utils import get_readable_timestamp
from metrics import accuracy


def train(model, train_loader, optimizer, epoch_id=0, scheduler=None, device="cpu",
          autosave_period=None, valid_period=None, test_routine=None,
          test_loader=None):
    tboard_writer = SummaryWriter()
    model.train()

    with tqdm(total=len(train_loader) * train_loader.batch_size,
              desc=f'Epoch {epoch_id + 1}',
              unit='image') as pbar:
        for batch_idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss_function = CrossEntropyLoss(reduction="mean")
            train_loss = loss_function(output, target)
            train_loss.backward()
            optimizer.step()

            global_step = epoch_id * len(train_loader) + batch_idx
            tboard_writer.add_scalar("Loss/Train", train_loss.item(), global_step)

            if valid_period:
                if (batch_idx + 1) % valid_period == 0:
                    pred = output.argmax(dim=1, keepdim=True)
                    train_acc = accuracy(pred, target, norm=True)
                    tboard_writer.add_scalar("Accuracy/Train", train_acc, global_step)
                    assert callable(test_routine)
                    assert test_loader is not None
                    val_loss, result = test_routine(model, device, test_loader, batches=1)
                    if scheduler is not None:
                        scheduler.step(val_loss)
                        tboard_writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)
                    tboard_writer.add_scalar("Loss/Val", val_loss, global_step)
                    tboard_writer.add_scalar("Accuracy/Val", result["accuracy"], global_step)
                    if "images" in result.keys():
                        img_grid_pred = torchvision.utils.make_grid(result["images"])
                        tboard_writer.add_image("Predictions", img_tensor=img_grid_pred,
                                            global_step=global_step, dataformats='CHW')

            if autosave_period is not None:
                if (batch_idx + 1) % autosave_period == 0:
                    model_name = str(model) + "_" + get_readable_timestamp() + "_epoch_" + \
                               str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                    torch.save(model.state_dict(), model_name)
                    print(model_name, " has been saved")

            pbar.update(train_loader.batch_size)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch-train", type=int, default=32,
                        help="Size of batch for training")
    parser.add_argument("--batch-valid", type=int, default=8,
                        help="Size of batch for validation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Target device: cpu/cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Type of optmizer")
    parser.add_argument("--autosave-period", type=int, default=0,
                        help="Period of model autosave")
    parser.add_argument("--autosave-period-unit", type=str, default="e",
                        help="Units for autosave (e/b)")
    parser.add_argument("--valid-period", type=int, default=100,
                        help="Period of validation")
    parser.add_argument("--valid-period-unit", type=str, default="e",
                        help="Units for validation (e/b)")
    parser.add_argument("--pretrained", type=str,
                        help="Abs path to pretrained model")
    parser.add_argument("--scheduler", type=int, default=1,
                        help="Use lr scheduler or not")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = MnistClassifier()
    model = model.to(args.device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    train_dloader = get_train_dataloader(path="dataset", batch_size=args.batch_train)
    test_dloader = get_test_dataloader(path="dataset", batch_size=args.batch_valid)

    optimizer = None
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.learning_rate)

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    try:
        for e in range(args.epochs):
            train(model, train_dloader, optimizer, e, scheduler=scheduler, device=args.device,
                  autosave_period=None, valid_period=args.valid_period, test_routine=test,
                  test_loader=test_dloader)
        model_name = "pretrained_models/" + str(model) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(model.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
    except KeyboardInterrupt:
        model_name = "pretrained_models/" + str(model) + "_terminated_" + get_readable_timestamp() + ".pt"
        torch.save(model.state_dict(), model_name)
        print("Training terminated. Model " + model_name + " has been saved")
    return


if __name__ == "__main__":
    main()
