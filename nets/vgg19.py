import torch
import torch.nn.functional as F
from torchvision import models, transforms
from collections import namedtuple, OrderedDict


# Borrowed from https://github.com/GrumpyZhou/image-matching-toolbox

class Vgg19(torch.nn.Module):

    # modified from the original @ https://github.com/chenyuntc/pytorch-book/blob/master/chapter08-neural_style/PackedVGG.py

    def __init__(self, batch_normalization=True, required_layers=None):

        if required_layers == None and batch_normalization:
            required_layers = [3, 10, 17, 30, 43, 46]
        elif required_layers == None:
            required_layers = [2, 7, 12, 21, 30, 32]
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(Vgg19, self).__init__()

        if batch_normalization:
            features = list(models.vgg19_bn(pretrained=True).features)[:47]  # get vgg features
        else:
            features = list(models.vgg19(pretrained=True).features)[:33]  # get vgg features
            # features = list(models.resnet50(pretrained = True).children())[:-3] # get vgg features

        self.features = torch.nn.ModuleList(features).eval()  # construct network in eval mode

        for param in self.features.parameters():  # we don't need graidents, turn them of to save memory
            param.requires_grad = False

        self.required_layers = required_layers  # record required layers to save them in forward

        for layer in required_layers[:-1]:
            self.features[layer + 1].inplace = False  # do not overwrite in any layer

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)

        vgg_outputs = namedtuple("Outputs", ['conv1_2', 'conv2_2', 'relu3_2', 'relu4_2', 'conv5_2', 'conv5_3'])

        return vgg_outputs(*results)


class FineVgg19(torch.nn.Module):

    def __init__(self, batch_normalization=True, required_layers=None):

        if required_layers == None and batch_normalization:
            required_layers = [3]
        elif required_layers == None:
            required_layers = [2]
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(FineVgg19, self).__init__()

        if batch_normalization:
            features = list(models.vgg19_bn(pretrained=True).features)[:4]  # get vgg features
        else:
            features = list(models.vgg19(pretrained=True).features)[:3]  # get vgg features
            # features = list(models.resnet50(pretrained = True).children())[:-3] # get vgg features

        self.features = torch.nn.ModuleList(features).eval()  # construct network in eval mode

        for param in self.features.parameters():  # we don't need graidents, turn them of to save memory
            param.requires_grad = False

        self.required_layers = required_layers  # record required layers to save them in forward

        for layer in required_layers[:-1]:
            self.features[layer + 1].inplace = False  # do not overwrite in any layer

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)

        vgg_outputs = namedtuple("Outputs", ['conv1_2'])

        return vgg_outputs(*results)


class CoarseVgg19(torch.nn.Module):

    def __init__(self, batch_normalization=True, required_layers=None):

        if required_layers == None and batch_normalization:
            required_layers = [10, 17, 30, 43, 46]
        elif required_layers == None:
            required_layers = [7, 12, 21, 30, 32]
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(CoarseVgg19, self).__init__()

        if batch_normalization:
            features = list(models.vgg19_bn(pretrained=True).features)[:47]  # get vgg features
        else:
            features = list(models.vgg19(pretrained=True).features)[:33]  # get vgg features

        self.features = torch.nn.ModuleList(features).eval()  # construct network in eval mode

        for param in self.features.parameters():  # we don't need graidents, turn them of to save memory
            param.requires_grad = False

        self.required_layers = required_layers  # record required layers to save them in forward

        for layer in required_layers[:-1]:
            self.features[layer + 1].inplace = False  # do not overwrite in any layer

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)

        vgg_outputs = namedtuple("Outputs", ['conv2_2', 'relu3_2', 'relu4_2', 'conv5_2', 'conv5_3'])

        return vgg_outputs(*results)
