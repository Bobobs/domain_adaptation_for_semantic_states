"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import pdb

import os
from glob import glob
import pandas as pd
from PIL import Image
import copy
from sklearn.model_selection import StratifiedShuffleSplit


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from datetime import datetime

import utils
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.alignment.bsp import BatchSpectralPenalizationLoss, ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

#from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    exp_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Data loading code

    #source_PATH = "/home/gtzelepis/data_semantic_scene/source_domain/simulated_for_robotics/"
    source_PATH = "/home/gtzelepis/data_semantic_scene/source_domain/human_for_robot/human_demos_to_human_demos/"
    domain_PATH_train = "/home/gtzelepis/data_semantic_scene/target_domain/cropped_robotic/"
    #domain_PATH_test = "/home/gtzelepis/data_semantic_scene/target_domain/robotic_target_test/orange/release_grasp_orange/"
    domain_PATH_test = "/home/gtzelepis/data_semantic_scene/target_domain/robotic_target_test/pink"
    #domain_PATH_test = "/home/gtzelepis/cropped_UR/domain_test/"
    #domain_PATH_test = "/home/gtzelepis/cropped_shirts/UR/lift_shirt_black/"
    #domain_PATH_test = "/home/gtzelepis/cropped_shirts/kinova/release_grasp_shirt_black"
    EXT = "*.csv"
    img_EXT = "*.png"

    if source_PATH.endswith("robotics/"):
        datasource = "robotics"
    else:
        datasource = "human_demos"

    def get_frame(PATH):
        all_csv_files = []
        all_img_files = []
        for path, subdir, files in os.walk(PATH):
            for file in glob(os.path.join(path, EXT)):
                if file.endswith('data.csv'):
                    all_csv_files.append(file)
            for i in glob(os.path.join(path, img_EXT)):
                all_img_files.append(i)

        li = []

        for filename in all_csv_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        return frame, all_img_files, all_csv_files

    source_frame, all_img_files_source, all_csv_files_source = get_frame(source_PATH)
    domain_train_frame, all_img_files_domain_train, all_csv_files_domain_train = get_frame(domain_PATH_train)
    domain_test_frame, all_img_files_domain_test, all_csv_files_domain_test = get_frame(domain_PATH_test)

    states = source_frame.cloth_state.unique()

    def get_idx(frame, states):
        state = states
        state2idx = dict((state, idx) for idx, state in enumerate(state))
        idx2state = dict((idx, state) for idx, state in enumerate(state))
        frame['label_idx'] = [state2idx[b] for b in frame.cloth_state]
        return

    get_idx(source_frame, states)
    get_idx(domain_train_frame, states)
    get_idx(domain_test_frame, states)



        
    class CustomDataset(Dataset):
        def __init__(self, labels_df, img_path, transform=None):
            self.labels_df = labels_df
            self.img_path = img_path
            self.transform = transform

        def __len__(self):
            return self.labels_df.shape[0]

        def __getitem__(self, idx):
            image_name = self.labels_df.filename[idx]
            for file in self.img_path:
                if file.endswith(image_name + '.png'):
                    img = Image.open(file)
            #                 print(file)
            label = self.labels_df.label_idx[idx]
            #         print(file)

            if self.transform:
                img = self.transform(img)

            return img, label

    input_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # these vectors is the mean and the std from the statistics in imagenet. They are always the same as far as I can recall
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            #         transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
    train_split_idx, val_split_idx = next(iter(stratified_split.split(source_frame.filename, source_frame.cloth_state)))
    train_df = source_frame.iloc[train_split_idx].reset_index()
    val_df = source_frame.iloc[val_split_idx].reset_index()

    source_train_dataset = CustomDataset(train_df, all_img_files_source, transform=data_transforms['train'])
    val_dataset = CustomDataset(val_df, all_img_files_source, transform=data_transforms['val'])
    target_train_dataset = CustomDataset(domain_train_frame, all_img_files_domain_train,
                                         transform=data_transforms['train'])
    test_dataset = CustomDataset(domain_test_frame, all_img_files_domain_test, transform=data_transforms['val'])

    train_source_loader = DataLoader(source_train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    train_target_loader = DataLoader(target_train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    num_classes = 9
    #


    # train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
    #                                             random_horizontal_flip=not args.no_hflip,
    #                                             random_color_jitter=False, resize_size=args.resize_size,
    #                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    # val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
    #                                         norm_mean=args.norm_mean, norm_std=args.norm_std)
    # print("train_transform: ", train_transform)
    # print("val_transform: ", val_transform)
    #
    # train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
    #     utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    # train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    bsp_penalty = BatchSpectralPenalizationLoss().to(device)

    # resume from the best checkpoint
    # if args.phase != 'train':
    #     checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    #     classifier.load_state_dict(checkpoint)
    #
    # # analysis the model
    # if args.phase == 'analysis':
    #     # extract features from both domains
    #     feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
    #     source_feature = collect_feature(train_source_loader, feature_extractor, device)
    #     target_feature = collect_feature(train_target_loader, feature_extractor, device)
    #     # plot t-SNE
    #     tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
    #     tsne.visualize(source_feature, target_feature, tSNE_filename)
    #     print("Saving t-SNE to", tSNE_filename)
    #     # calculate A-distance, which is a measure for distribution discrepancy
    #     A_distance = a_distance.calculate(source_feature, target_feature, device)
    #     print("A-distance =", A_distance)
    #     return

    # if args.phase == 'test':
    #     acc1 = utils.validate(test_loader, classifier, args, device)
    #     print(acc1)
    #     return

    # if args.pretrain is None:
    #     # first pretrain the classifier wish source data
    #     print("Pretraining the model on source domain.")
    #     args.pretrain = logger.get_checkpoint_path('pretrain')
    #     pretrain_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
    #                                      pool_layer=pool_layer, finetune=not args.scratch).to(device)
    #     pretrain_optimizer = SGD(pretrain_model.get_parameters(), args.pretrain_lr, momentum=args.momentum,
    #                              weight_decay=args.weight_decay, nesterov=True)
    #     pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
    #                                      lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
    #                                          -args.lr_decay))
    #     # start pretraining
    #     for epoch in range(args.pretrain_epochs):
    #         print("lr:", pretrain_lr_scheduler.get_lr())
    #         # pretrain for one epoch
    #         utils.empirical_risk_minimization(train_source_iter, pretrain_model, pretrain_optimizer,
    #                                           pretrain_lr_scheduler, epoch, args,
    #                                           device)
    #         # validate to show pretrain proces
    #         utils.validate(val_loader, pretrain_model, args, device)
    #
    #     torch.save(pretrain_model.state_dict(), args.pretrain)
    #     print("Pretraining process is done.")
    #
    # checkpoint = torch.load(args.pretrain, map_location='cpu')
    # classifier.load_state_dict(checkpoint)
    #
    # # start training
    # best_acc1 = 0.
    # for epoch in range(args.epochs):
    #     print("lr:", lr_scheduler.get_last_lr()[0])
    #     # train for one epoch
    #     train(train_source_iter, train_target_iter, classifier, domain_adv, bsp_penalty, optimizer,
    #           lr_scheduler, epoch, args)
    #
    #     # evaluate on validation set
    #     acc1 = utils.validate(val_loader, classifier, args, device)
    #
    #     # remember best acc@1 and save checkpoint
    #     torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
    #     if acc1 > best_acc1:
    #         shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
    #     best_acc1 = max(acc1, best_acc1)
    #
    # print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))
    load_path = "/home/gtzelepis/Transfer-Learning-Library/examples/domain_adaptation/image_classification/bsp/checkpoints/bestcuda:1"
    #load_path = "/home/gtzelepis/Transfer-Learning-Library/examples/domain_adaptation/image_classification/bsp/checkpoints/best9classes"
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path(load_path)))
    acc1, predictions, predictions2, predictions3 , predictions4, predictions5, predictions6, predictions7, predictions8, predictions9= utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    domain_test_frame['predictions'] = predictions.tolist()
    domain_test_frame['predictions2'] = predictions2.tolist()
    domain_test_frame['predictions3'] = predictions3.tolist()
    domain_test_frame['predictions4'] = predictions4.tolist()
    domain_test_frame['predictions5'] = predictions5.tolist()
    domain_test_frame['predictions6'] = predictions6.tolist()
    domain_test_frame['predictions7'] = predictions7.tolist()
    domain_test_frame['predictions8'] = predictions8.tolist()
    domain_test_frame['predictions9'] = predictions9.tolist()
    domain_test_frame.to_csv('UR_top9_pink.csv', index = False)


    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource

    domain_test_frame['predictions'] = predictions.tolist()
    domain_test_frame['predictions2'] = predictions2.tolist()
    domain_test_frame['predictions3'] = predictions3.tolist()
    domain_test_frame['predictions4'] = predictions4.tolist()
    domain_test_frame['predictions5'] = predictions5.tolist()
    domain_test_frame['predictions6'] = predictions6.tolist()
    domain_test_frame['predictions7'] = predictions7.tolist()
    domain_test_frame['predictions8'] = predictions8.tolist()
    domain_test_frame['predictions9'] = predictions9.tolist()
    domain_test_frame.to_csv('UR_top9.csv', index = False)

    # domain_test_frame.to_csv('kinova.csv', index = False)
    # p = figure(
    #     width=1000,
    #     height=1000,
    #     title="Task Stream",
    #     toolbar_location="above",
    #     x_axis_type="datetime",
    # )

    # p.rect(
    #     source=ColumnDataSource(domain_test_frame),
    #     x="index",
    #     y="predictions",
    #     width=0.01,
    #     height=0.4
    # )

    # #show(p)

    # logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, bsp_penalty: BatchSpectralPenalizationLoss,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        bsp_loss = bsp_penalty(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off + bsp_loss * args.trade_off_bsp

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BSP for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain checkpoint for classification model')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off-bsp', default=2e-4, type=float,
                        help='the trade-off hyper-parameter for bsp loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--pretrain-lr', default=0.001, type=float, help='initial pretrain learning rate')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrain-epochs', default=3, type=int, metavar='N',
                        help='number of total epochs(pretrain) to run (default: 3)')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='bsp',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
