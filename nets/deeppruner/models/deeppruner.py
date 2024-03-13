# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
from nets.deeppruner.models.submodules3d import MinDisparityPredictor, MaxDisparityPredictor, \
    CostAggregator
from nets.deeppruner.models.submodules2d import RefinementNet
from nets.deeppruner.models.submodules import SubModule, conv_relu, convbn_2d_lrelu, \
    convbn_3d_lrelu
from nets.deeppruner.models.utils import SpatialTransformer, UniformSampler
from nets.deeppruner.models.patch_match import PatchMatch
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.deeppruner.models.config import config as args


class DeepPruner(SubModule):
    def __init__(self):
        super(DeepPruner, self).__init__()
        self.scale = args.cost_aggregator_scale
        self.max_disp = args.max_disp // self.scale
        self.mode = args.mode

        self.patch_match_args = args.patch_match_args
        self.patch_match_sample_count = self.patch_match_args.sample_count
        self.patch_match_iteration_count = self.patch_match_args.iteration_count
        self.patch_match_propagation_filter_size = self.patch_match_args.propagation_filter_size

        self.post_CRP_sample_count = args.post_CRP_sample_count
        self.post_CRP_sampler_type = args.post_CRP_sampler_type
        hourglass_inplanes = args.hourglass_inplanes

        #   refinement input features are composed of:
        #                                       left image low level features +
        #                                       CA output features + CA output disparity

        if self.scale == 8:
            from nets.deeppruner.models.feature_extractor_fast import feature_extraction
            refinement_inplanes_1 = args.feature_extractor_refinement_level_1_outplanes + 1
            self.refinement_net1 = RefinementNet(refinement_inplanes_1)
        else:
            from feature_extractor_best import feature_extraction

        refinement_inplanes = args.feature_extractor_refinement_level_outplanes + self.post_CRP_sample_count + 2 + 1
        self.refinement_net = RefinementNet(refinement_inplanes)

        # cost_aggregator_inplanes are composed of:  
        #                            left and right image features from feature_extractor (ca_level) + 
        #                            features from min/max predictors + 
        #                            min_disparity + max_disparity + disparity_samples

        cost_aggregator_inplanes = 2 * (args.feature_extractor_ca_level_outplanes +
                                        self.patch_match_sample_count + 2) + 1
        self.cost_aggregator = CostAggregator(cost_aggregator_inplanes, hourglass_inplanes)

        self.feature_extraction = feature_extraction()
        self.min_disparity_predictor = MinDisparityPredictor(hourglass_inplanes)
        self.max_disparity_predictor = MaxDisparityPredictor(hourglass_inplanes)
        self.spatial_transformer = SpatialTransformer()
        self.patch_match = PatchMatch(self.patch_match_propagation_filter_size)
        self.uniform_sampler = UniformSampler()

        # Confidence Range Predictor(CRP) input features are composed of:  
        #                            left and right image features from feature_extractor (ca_level) + 
        #                            disparity_samples

        CRP_feature_count = 2 * args.feature_extractor_ca_level_outplanes + 1
        self.dres0 = nn.Sequential(convbn_3d_lrelu(CRP_feature_count, 64, 3, 1, 1),
                                   convbn_3d_lrelu(64, 32, 3, 1, 1))

        self.dres1 = nn.Sequential(convbn_3d_lrelu(32, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, hourglass_inplanes, 3, 1, 1))

        self.min_disparity_conv = conv_relu(1, 1, 5, 1, 2)
        self.max_disparity_conv = conv_relu(1, 1, 5, 1, 2)
        self.ca_disparity_conv = conv_relu(1, 1, 5, 1, 2)

        self.ca_features_conv = convbn_2d_lrelu(self.post_CRP_sample_count + 2,
                                                self.post_CRP_sample_count + 2, 5, 1, 2, dilation=1, bias=True)
        self.min_disparity_features_conv = convbn_2d_lrelu(self.patch_match_sample_count + 2,
                                                           self.patch_match_sample_count + 2, 5, 1, 2, dilation=1,
                                                           bias=True)
        self.max_disparity_features_conv = convbn_2d_lrelu(self.patch_match_sample_count + 2,
                                                           self.patch_match_sample_count + 2, 5, 1, 2, dilation=1,
                                                           bias=True)

        self.weight_init()

    def forward(self, left_input, right_input):
        if self.scale == 8:
            left_spp_features, left_low_level_features, left_low_level_features_1 = self.feature_extraction(left_input)
            right_spp_features, right_low_level_features, _ = self.feature_extraction(
                right_input)
        else:
            left_spp_features, left_low_level_features = self.feature_extraction(left_input)
            right_spp_features, right_low_level_features = self.feature_extraction(right_input)

        return left_spp_features, left_low_level_features, right_spp_features, right_low_level_features


if __name__ == '__main__':
    from torch.autograd import Variable

    model = DeepPruner()
    imgL = torch.rand(1, 3, 255, 255)
    imgR = torch.rand(1, 3, 255, 255)
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    # print(fe)
    # print(fe(imgL))
    # print(result1.shape)
    # print(rs2.shape)
    # print(rs3.shape)
    # print(rs4.shape)
    model.eval()
    # print(model)
    load_path = r"D:\hy\Pycharm\IME\Algorithms\DFM\python\models\off-the-shelf\DeepPruner\DeepPruner-fast-kitti.tar"
    state_dict = torch.load(load_path)

    model.load_state_dict(state_dict['state_dict'], strict=False)
    fe = model.feature_extraction
    f_result1 = fe(imgL)
