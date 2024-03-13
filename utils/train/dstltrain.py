# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import torch
import torch.optim as optim

from tools import common, distillation_trainer
from tools.dataloader import *
from nets.patchnet import *
from nets.resnet_r2d2 import ResNet18_R2D2
from nets.mobile_r2d2 import MobileNetV3_R2D2
from nets.losses import *
from nets.distillation_feature_loss import *

default_teacher_net = "Quad_L2Net_ConfCFS()"

default_student_net = "Student_Quad_L2Net_ConfCFS()"

toy_db_debug = """SyntheticPairDataset(
    ImgFolder('imgs'), 
            'RandomScale(256,1024,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""

db_web_images = """SyntheticPairDataset(
    web_images, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_images = """SyntheticPairDataset(
    aachen_db_images, 
        'RandomScale(256,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_style_transfer = """TransformedPairs(
    aachen_style_transfer_pairs,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_flow = "aachen_flow_pairs"

data_sources = dict(
    D=toy_db_debug,
    W=db_web_images,
    A=db_aachen_images,
    F=db_aachen_flow,
    S=db_aachen_style_transfer,
)

default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(256,1024,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(192)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""

# default_loss = """MultiLoss(
#         1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
#         1, CosimLoss(N=`N`),
#         1, PeakyLoss(N=`N`))"""


default_distillation_loss = """MultiLoss(1, DistillationLoss())"""


def load_network(model_fn):
    print(model_fn)
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class MyTrainer(distillation_trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """

    def forward_backward(self, inputs):
        # print(inputs)
        x = [inputs.pop('img1'), inputs.pop('img2')]
        # print(x[0].shape)
        student_output = self.student_net(imgs=x)
        with torch.no_grad():
            teacher_output = self.teacher_net(imgs=x)
        # print(dict(inputs, **student_output))
        # print("--"*20)

        # print(type(teacher_output['repeatability'][0]))
        # print(teacher_output['repeatability'])
        allvars = dict(inputs, **student_output)
        allvars = dict(allvars, **teacher_output)
        # print(allvars.keys())
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details


if __name__ == '__main__':
    import argparse

    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser("Train R2D2")

    parser.add_argument("--data-loader", type=str, default=default_dataloader)
    parser.add_argument("--train-data", type=str, default=list('WASF'), nargs='+',
                        choices=set(data_sources.keys()))
    parser.add_argument("--teacher_net", type=str, default=default_teacher_net, help='teacher network architecture')
    parser.add_argument("--student_net", type=str, default=default_student_net, help='student network architecture')

    parser.add_argument("--model", type=str, default="models/r2d2_WASF_N16.pt",
                        help='pretrained model path')
    parser.add_argument("--save-path", type=str, default=r"results\R2D2_justDistillation\dstl_", help='model save_path path')

    parser.add_argument("--loss", type=str, default=default_distillation_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--N", type=int, default=16, help="patch size for repeatability")

    parser.add_argument("--epochs", type=int, default=25, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=2, help="batch size")
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)

    parser.add_argument("--threads", type=int, default=2, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')

    args = parser.parse_args()

    iscuda = common.torch_set_gpu(args.gpu)
    common.mkdir_for(args.save_path)

    # Create data loader
    from datasets import *

    db = [data_sources[key] for key in args.train_data]
    print("data_loader=" + args.data_loader)
    print("db=" + str(db))
    ans = args.data_loader.replace('`data`', ','.join(db)).replace('\n', '')
    db = eval(ans)
    print("Training image database =", db)
    loader = threaded_loader(db, iscuda, args.threads, args.batch_size, shuffle=True)

    # create network
    print("\n>> Creating student net = " + args.student_net)
    student_net = eval(args.student_net)
    print(f" ( Model size: {common.model_size(student_net) / 1000:.0f}K parameters )")
    # print("\n>> Creating teacher net = " + args.teacher_net)
    teacher_net = load_network(args.model).eval()


    # print(f" ( Model size: {common.model_size(teacher_net) / 1000:.0f}K parameters )")

    # initialization
    # if args.pretrained:
    #     checkpoint = torch.load(args.pretrained, lambda a,b:a)
    #     teacher_net.load_pretrained(checkpoint['state_dict'])

    # create losses
    loss = args.loss.replace('`sampler`', args.sampler).replace('`N`', str(args.N))
    print("\n>> Creating loss = " + loss)
    loss = eval(loss.replace('\n', ''))

    # create optimizer
    optimizer = optim.Adam([p for p in student_net.parameters() if p.requires_grad],
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    train = MyTrainer(student_net, teacher_net, loader, loss, optimizer)
    if iscuda: train = train.cuda()

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n>> Starting epoch {epoch}...")
        train()
        # print(train())
    # train()
        print(f"\n>> Saving model to {args.save_path}")
        torch.save({'net': args.student_net, 'state_dict': student_net.state_dict()}, args.save_path+f"{epoch}.pt")






