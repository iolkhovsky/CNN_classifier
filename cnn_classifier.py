import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.bn1 = nn.BatchNorm2d(256)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = relu(self.bn1(x))
        x = self.conv2(x)
        x = relu(self.bn2(x))
        x = max_pool2d(x, 2)
        x = self.conv3(x)
        x = relu(self.bn3(x))
        x = self.conv4(x)
        x = relu(self.bn4(x))
        x = max_pool2d(x, 2)
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*4*4, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.do2 = torch.nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.do1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.fc2(x)
        return x


class MnistClassifier(nn.Module):

    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.fext = FeatureExtractor()
        self.clf = Classifier()
        self.act = nn.Softmax(dim=1)
        return

    def forward(self, x):
        x = self.fext(x)
        logits = self.clf(x)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        probs = self.act(logits)
        return probs

    def __str__(self):
        return "CnnClassifierMNIST"
