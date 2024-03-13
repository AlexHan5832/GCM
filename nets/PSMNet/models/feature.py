from __future__ import print_function

from collections import namedtuple

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from nets.PSMNet.models.submodule import *


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        result_1, result_2 = self.feature_extraction(x)

        rs3 = F.interpolate(result_2, scale_factor=0.5, mode='nearest')
        # rs4 = F.interpolate(result_2, scale_factor=0.5, mode='nearest')
        outputs = namedtuple("Outputs", ['layer1', 'feature', "f2"])

        return outputs(result_1, result_2, rs3)

if __name__ == '__main__':
    model = PSMNet(192)
    # PATH = r"D:\hy\Pycharm\PSMNet\models\off\test_state_dict.pth"
    # model.load_state_dict(torch.load(PATH))

    # 打印模型的状态字典
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    img = torch.rand(1,3,255,300)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
    print(outputs[2].shape)




