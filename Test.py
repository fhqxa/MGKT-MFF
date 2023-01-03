import math
import os
import argparse
import torch
from matplotlib import pyplot as plt
import Utils
from Dataloader import tree_dataloader
import os.path as osp
import random
import torch.nn as nn
import numpy as np
from Dataloader.Flat_dealdataset import dealdataset
from Utils import losses, ratio
from Models import resnet32_hier, resnet32_flat,  smffresnet_flat, smffresnet_hier
from torch.utils.data.sampler import WeightedRandomSampler
from Utils import util
from torchvision import transforms


class NetworkManager(object):
    def __init__(self, options, path, args):
        self.options = options
        self.path = path
        self.device = options['device']
        self.args = args
        print('Starting to prepare network and data...')
        self.net = self._net_choice(self).to(self.device)  # 调用网络
    

        self.net.load_state_dict('ckpt.best.pth')
        print('Network is as follows:')
        self.test_data = tree_cifar_Hier.DatasetLoader('test', return_path=False,
                                                       data_dir=self.path["test_data_path"])
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=100, shuffle=False, num_workers=4,
                                                       pin_memory=True)

    def _accuracy_hier(self):
        with torch.no_grad():
            self.net.eval()
            num_acc_c = 0
            num_acc_f = 0
            num_total = 0
            for imgs, label_test_f, label_test_c, in self.test_loader:
                # print('imgs',imgs.shape,label_test_c,label_test_f)
                imgs, label_test_f, label_test_c = imgs.to(self.device), label_test_f.to(self.device), label_test_c.to(
                    self.device)
                out_c, out_f = self.net(imgs)
                _, pred_c = torch.max(out_c, dim=1)
                _, pred_f = torch.max(out_f, dim=1)
                num_acc_c += torch.sum(pred_c == label_test_c)
                num_acc_f += torch.sum(pred_f == label_test_f)
                num_total += label_test_f.size(0)

                # -------------------------------------Compute TIE and FH-----------------------------------------

                f = open(  'CSMFF_MG_None_cifar50.csv', 'a')
                for i in range(100):
                    f.write(str(label_test_f[i].item()) + ',' + str(pred_f[i].item()) + '\n')
            print(num_acc_f.detach().cpu().numpy() * 100 / num_total)
        return num_acc_f.detach().cpu().numpy() * 100 / num_total

    def _accuracy_flat(self):
        num_acc = 0
        num_total = 0
        self.net.eval()
        TIE = []
        FH = []
        with torch.no_grad():
            for imgs, label_test_f, label_test_c in self.test_loader:  
                imgs, label_test_f, label_test_c = imgs.to(self.device), label_test_f.to(self.device), label_test_c.to(
                    self.device)
                output = self.net(imgs)
                output = torch.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                num_acc += torch.sum(pred == label_test_f.detach_())
                num_total += label_test_f.size(0)

                # -------------------------------------Compute TIE and FH-----------------------------------------

                f = open(
                    'cifar_resnet32_CE_None_exp_200_0/35.50',
                    'a')
                for i in range(100):
                    f.write(str(label_test_f[i].item()) + ',' + str(pred[i].item()) + '\n')
            print(num_acc.detach().cpu().numpy() * 100 / num_total)
        return num_acc.detach().cpu().numpy() * 100 / num_total

    def mkdir(self, path):
        import os
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")

        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return True
        else:
            return False

    @staticmethod
    def _net_choice(self):
        if self.args.Hier_dataset:
            if self.args.flag == 'resnet_hier':
                print('resnet_hier')
                base_model = resnet32_hier.resnet32(self.args.num_classes_c, self.args.num_classes_f, use_norm=False)
            elif self.args.flag == 'smffresnet_hier':
                print("smffresnet_hier")
                base_model = smffresnet_hier.resnet32(self.args.num_classes_c, self.args.num_classes_f)
            print(base_model.linear_f)
        else:
            if self.args.flag == 'resnet_flat':
                base_model = resnet32_flat.resnet32(self.args.num_classes_f, use_norm=False)
                print('resnet_flat')
            elif self.args.flag == 'smffresnet_flat':
                print("smffresnet_flat")
                base_model = smffresnet_flat.resnet32(self.args.num_classes_f)

        return base_model


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical classification experiments under long-tailed datasets'
    )
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')

    parser.add_argument('--epochs', type=int, default=200, help='batch size for training')
    parser.add_argument('--num_classes_c', type=int, default=20, help='all unique coarse classes number')
    parser.add_argument('--num_classes_f', type=int, default=100, help='all unique fine classes number')

    parser.add_argument('--flag', type=str, default='resnet_flat',
                        help='Hier_dataset is True = resnet_hier/resnetmff_hier/smffresnet_hier    false = resnet_flat/resnetmff_flat/smffresnet_flat')
    # help='0：Hier_dataset is True =ResNet32_Hier else =ResNet32_Flat, 1:ResNet_MFF_Hier/ResNet_MFF_Flat')
    parser.add_argument('--Hier_dataset', type=bool, default=False, help='Whether is hierarchical structure')

    parser.add_argument('--Update_hier', type=bool, default=False, help='Whether to re-establish the tree structure')
    parser.add_argument('--ratio', type=int, default=200, help='unblance training')
    parser.add_argument('--dataset', type=int, default=100,
                        help='set dataset voc20 or cifar100 or tiredImagenet608 or Sun324')
    parser.add_argument('--gpu_id', type=int, default=0, help='choose one gpu for training')
    parser.add_argument('--img_size', type=int, default=32, help='image\'s size for transforms')

    parser.add_argument('--warm', type=int, default=210, help='Controal oversampling, using flat state')
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--dataset_name', type=str, default='test',
                        help='in order to create fold ')
    args = parser.parse_args()
    assert args.gpu_id.__class__ == int
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.Update_hier:
        ratio.CreateImageDirectory(isCorse=True, long_tail=True, isChinese=True, ratio_longtail=args.ratio)
    if args.dataset == 100:
        if args.ratio == 10:
            train_root_data_path = '/cifar_100_long_tail_fen_ratio_10'
        elif args.ratio == 20:
            train_root_data_path = '/cifar_100_long_tail_fen_ratio_20'
        elif args.ratio == 50:
            train_root_data_path = '/cifar_100_long_tail_fen_ratio_50'
        elif args.ratio == 100:
            train_root_data_path = '/cifar_100_long_tail_fen_ratio_100'
        elif args.ratio == 200:
            train_root_data_path = '/cifar_100_long_tail_fen_ratio_200'
        else:
            train_root_data_path = '/cifar_100_long_tail_fen_blance'
        test_root_data_path = "/public_Hier_test"
    elif args.dataset == 20:
        if args.ratio == 57:
            train_root_data_path = '/VOC2012_Hier'
            test_root_data_path = '/VOC2012_Hier'
    elif args.dataset == 324:
        if args.ratio == 28:
            train_root_data_path = '/SUN_datasets/Sun_hier'
            test_root_data_path = '/SUN_datasets/Sun_hier'
    elif args.dataset == 608:
        if args.ratio == 203:
            train_root_data_path = "/TiredImagenet_datasets/TiredImagnet_hier"
            test_root_data_path = "/TiredImagenet_datasets/TiredImagnet_hier"
    options = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'dataset': args.dataset,
        'device': torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu'),
        'seed': args.seed,
        'warm': args.warm
    }
    path = {
        'test_data_path': test_root_data_path,
    }
    manager = NetworkManager(options, path, args)  # manager为类的对象
    if args.Hier_dataset:
        manager._accuracy_hier()  # 开始训练，trainself.options['net_choice']()为trainer
    else:
        manager._accuracy_flat()


if __name__ == '__main__':
    main()
