import torch
from torch.nn import CrossEntropyLoss
from metrics import accuracy


def test(model, device, test_loader, batches=None):
    model.eval()
    test_loss = 0
    res = {}
    correct = 0
    loss_function = CrossEntropyLoss(reduction='sum')
    batch_size = test_loader.batch_size
    with torch.no_grad():
        for idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            acc = accuracy(pred, target, norm=False)
            correct += acc

            if batches is not None:
                if idx >= batches - 1:
                    break

        if batches is not None:
            test_loss = test_loss / (batches * batch_size)
            res["accuracy"] = correct / (batches * batch_size)
        else:
            test_loss = test_loss / (len(test_loader) * batch_size)
            res["accuracy"] = correct / (len(test_loader) * batch_size)

    return test_loss, res
