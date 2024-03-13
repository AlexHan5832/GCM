from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.CREStereo.nets.extractor import BasicEncoder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/raft.py
class CREStereo(nn.Module):
    def __init__(self, max_disp=192, downsample=False, test_mode=False):
        super(CREStereo, self).__init__()

        self.max_flow = max_disp
        self.downsample = downsample
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)


    def forward(self, x):

        # run the feature network
        fmap_dw2, fmap_dw4 = self.fnet(x)

        if self.downsample == True:

            fmap_dw8 = F.avg_pool2d(fmap_dw4, 2, stride=2)
            # fmap_dw8 = F.interpolate(fmap_dw4, scale_factor=0.5, mode='nearest')

            outputs = namedtuple("Outputs", ['fmap_dw2', 'fmap_dw4', "fmap_dw8"])
            return outputs(fmap_dw2, fmap_dw4, fmap_dw8)
        else:
            outputs = namedtuple("Outputs", ['fmap_dw2', 'fmap_dw4'])
            return outputs(fmap_dw2, fmap_dw4)


if __name__ == '__main__':
    model = CREStereo(max_disp=256, downsample=True, test_mode=True)
    model_path = r'D:\hy\Pycharm\CREStereo-Pytorch\models\feat.pth'
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to("cpu")
    model.eval()

    x = torch.rand(1, 3, 300, 400)
    rs = model(x)
    for element in rs:
        print(element.shape)