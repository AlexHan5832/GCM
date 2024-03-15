import torch
import torch.nn.functional as F
from torchvision import models, transforms
from collections import namedtuple, OrderedDict


class Resnet50(torch.nn.Module):

    def __init__(self, required_layers=None):

        super(Resnet50, self).__init__()

        if required_layers is None:
            required_layers = [2, 4, 5, 6]

        features = list(models.resnet50(pretrained=True).children())[:-1]  # get ResNet children

        self.features = torch.nn.ModuleList(features).eval()  # construct network in eval mode
        # print(self.features)

        for param in self.features.parameters():  # we don't need graidents, turn them of to save memory
            param.requires_grad = False

        self.required_layers = required_layers  # record required layers to save them in forward

        for layer in required_layers[:-1]:
            self.features[layer+1].inplace = False  # do not overwrite in any layer

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)

        vgg_outputs = namedtuple("Outputs", ['l2_relu', 'l4', 'l5', 'l6'])

        return vgg_outputs(*results)

# if __name__ == '__main__':
#     model = Resnet50().to("cpu").eval()
#     x = torch.ones(1,3,250,300)
#     rs = model(x)
#     for e in rs:
#         print(e.shape)