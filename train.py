import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from utils import get_readable_timestamp


def train(model, train_loader, optimizer, device="cpu", epoch_id=None,
          autosave_period=None, valid_period=None, test_routine=None,
          test_loader=None):
    tboard_writer = SummaryWriter()
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss_function = CrossEntropyLoss()
        train_loss = loss_function(output, target)
        train_loss.backward()
        optimizer.step()

        global_step = epoch_id * len(train_loader) + batch_idx
        tboard_writer.add_scalar("Loss/Train", train_loss.item(), global_step)

        if valid_period is not None:
            assert callable(test_routine)
            assert test_loader is not None
            val_loss, result = test_routine(model, device, test_loader)
            tboard_writer.add_scalar("Loss/Val", val_loss.item(), global_step)
            tboard_writer.add_scalar("Accuracy", result["accuracy"])
            img_grid_pred = torchvision.utils.make_grid(result["images"])
            tboard_writer.add_image("Predictions", img_tensor=img_grid_pred,
                                    global_step=global_step, dataformats='CHW')

        if autosave_period is not None:
            if (batch_idx + 1) % autosave_period == 0:
                model_name = str(model) + "_" + get_readable_timestamp() + "_epoch_" + \
                           str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                torch.save(model.state_dict(), model_name)
                print(model_name, " has been saved")
    return

