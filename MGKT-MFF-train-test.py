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
from Models import resnet32_hier,  resnet32_flat, smffresnet_flat, smffresnet_hier
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
        print('Network is as follows:')
        self.solver = torch.optim.SGD(  # 优化器
            self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'],
            weight_decay=self.options['weight_decay']
        )

        self.train_data = tree_cifar_Hier.DatasetLoader('train', return_path=False,
                                                        data_dir=self.path["train_data_path"])
        self.test_data = tree_cifar_Hier.DatasetLoader('test', return_path=False,
                                                       data_dir=self.path["test_data_path"])
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=100, shuffle=False, num_workers=4,
                                                       pin_memory=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.options['batch_size'],
                                                        shuffle=True, num_workers=4, pin_memory=True)
        self.weight = self.per_image_weight()
        self.num_sample = len(self.weight)
        self.train_loader_oversample = torch.utils.data.DataLoader(self.train_data,
                                                        sampler=WeightedRandomSampler(weights=self.weight,
                                                        num_samples=self.num_sample,
                                                         replacement=True), num_workers=4,
                                                        pin_memory=True, batch_size=self.options['batch_size'])
        self.tree = self.arraytotree()

    def get_lambda(self, cur_epoch, coarse_train_ep, fine_train_ep):
        if cur_epoch > self.options['epochs'] - fine_train_ep - 1:  # 30-15-1 =14
            my_lambda = 0
        elif cur_epoch < coarse_train_ep:  # 5
            my_lambda = 1
        else:  # 1-((11-5)/(30-15))**2 =1-0.4*0.4 =0.84  这个my_lambda 逐渐变小，直到变成0
            my_lambda = 1 - ((cur_epoch + 1 - coarse_train_ep) / (
                        self.options['epochs'] - fine_train_ep - coarse_train_ep)) ** 2
        return my_lambda

    @staticmethod
    def get_scalingfac(num1, num2):
        s1 = int(math.floor(math.log10(num1)))
        s2 = int(math.floor(math.log10(num2)))
        scale = 10 ** (s1 - s2)
        return scale

    def per_class_num(self):
        THE_PATH = os.path.join(self.path["train_data_path"], 'train')
        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path
        # print(coarse_folders)
        fine_folders = [os.path.join(coarse_label, label) \
                        for coarse_label in coarse_folders \
                        if os.path.isdir(coarse_label) \
                        for label in os.listdir(coarse_label)
                        ]
        per_num_class_fine = []
        per_num_class_coarse = []
        for path in fine_folders:
            a = len(os.listdir(path))
            per_num_class_fine.append(a)
        for path in coarse_folders:
            b = len(os.listdir(path))
            per_num_class_coarse.append(b)
        return per_num_class_coarse, per_num_class_fine

    # DRS computer
    def per_image_weight(self):
        THE_PATH = os.path.join(self.path["train_data_path"], 'train')
        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path
        # print(coarse_folders)
        fine_folders = [os.path.join(coarse_label, label) \
                        for coarse_label in coarse_folders \
                        if os.path.isdir(coarse_label) \
                        for label in os.listdir(coarse_label)
                        ]
        weight_fine = []  # per a img weight
        for path in fine_folders:
            num_per_class = len(os.listdir(path))
            num_per_class_fine = 1 / num_per_class
            for i in range(num_per_class):
                weight_fine.append(num_per_class_fine)
                #fprint(weight_fine)
        return weight_fine

    # tree struction
    def arraytotree(self):
        THE_PATH = os.path.join(self.path["train_data_path"], 'train')
        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, coarse_label))]  # coarse class path
        coarse_list = []
        for i, coarse in enumerate(coarse_folders):
            for fine in range(len(os.listdir(coarse))):
                coarse_list.append(i)
        tree = util.create_array(coarse_list)
        return tree

    def train_hier(self):
        self.mkdir(self.path['path_cur'])
        self.mkdir(self.path['path_model'])
        f = open(self.path['path_log'], 'w')
        f.write('basic information parameter:' + str(self.args.flag) + '\t\t' + 'ratio:'
                + str(self.args.ratio) + '\t\t' + 'seed:' + str(self.args.seed) + '\t\t' + 'loss_type:'
                + str(self.args.loss_type) + '\t\t' + 'train_rule:' + str(self.args.train_rule) + '\n')
        f.write(
            'Epoch\t TrainLoss_CF\t TrainLoss_C\t TrainLoss_F\tTrainAcc_C\tTrainAcc_F\t ||  TestLoss_CF\t TestLoss_C\t TestLoss_F\tTestAcc_c\t TIE\t FH\t TestAcc_F\n')
        epochs = np.arange(1, self.options['epochs'] + 1)
        print('Training process starts:...')
        print(
            'Epoch\t TrainLoss_CF\t TrainLoss_C\t TrainLoss_F\tTrainAcc_C\tTrainAcc_F\t ||  TestLoss_CF\t TestLoss_C\t TestLoss_F\tTestAcc_c\t TIE\t FH\t TestAcc_F')
        print('_' * 150)
        best_acc = 0.0
        test_acc = list()
        train_acc = list()

        cls_num_list_c, cls_num_list_f = self.per_class_num()
        print('cls_num_list', cls_num_list_c)
        for epoch in range(self.options['epochs']):
            self.adjust_learning_rate(self.solver, epoch, self.args)
            train_loss_epoch = list()
            train_loss_epoch_c = list()
            train_loss_epoch_f = list()
            num_correct_c = 0
            num_correct_f = 0
            num_total = 0
            my_lambda = self.get_lambda(epoch, coarse_train_ep=self.args.coarse_ep, fine_train_ep=self.args.fine_ep)
            if self.args.train_rule == 'None':
                # train_sampler = None
                per_cls_weights_c = None
                per_cls_weights_f = None
            elif self.args.train_rule == 'RW':
                per_cls_weights = 1 / np.array(cls_num_list_f)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            elif self.args.train_rule == 'DRW':
                if epoch < 160:
                    per_cls_weights_c = None
                    per_cls_weights_f = None
                else:
                    per_cls_weights_c = 1 / np.array(cls_num_list_c)
                    per_cls_weights_c = per_cls_weights_c / np.sum(per_cls_weights_c) * len(cls_num_list_c)
                    per_cls_weights_c = torch.FloatTensor(per_cls_weights_c).to(self.device)
                    per_cls_weights_f = 1 / np.array(cls_num_list_f)
                    per_cls_weights_f = per_cls_weights_f / np.sum(per_cls_weights_f) * len(cls_num_list_f)
                    per_cls_weights_f = torch.FloatTensor(per_cls_weights_f).to(self.device)
            elif self.args.train_rule == 'CB':
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list_f)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            elif self.args.train_rule == 'DCB':
                idx = epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list_f)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

            if self.args.loss_type == 'CE':
                self.criterion = nn.CrossEntropyLoss(weight=None).to(self.device)
            elif self.args.loss_type == 'Two_CE':
                self.criterion_c = nn.CrossEntropyLoss(weight=per_cls_weights_c).to(self.device)
                self.criterion_f = nn.CrossEntropyLoss(weight=per_cls_weights_f).to(self.device)
            elif self.args.loss_type == 'LDAM':
                self.criterion = losses.LDAMLoss(cls_num_list=cls_num_list_f, max_m=0.5, s=30,
                                                 weight=per_cls_weights).to(
                    self.device)
            elif self.args.loss_type == 'Two_LDAM':
                self.criterion_c = losses.LDAMLoss(cls_num_list=cls_num_list_c, max_m=0.5, s=30,
                                                   weight=per_cls_weights_c).to(self.device)
                self.criterion_f = losses.LDAMLoss(cls_num_list=cls_num_list_f, max_m=0.5, s=30,
                                                   weight=per_cls_weights_f).to(self.device)
            elif self.args.loss_type == 'Focal':
                self.criterion = losses.FocalLoss(per_cls_weights, gamma=1).to(self.device)
            self.net.train(True)  # TypeError: conv2d(): argument 'input' (position 1) must be Tensor, not bool
            for imgs, label_train_f, label_train_c, in self.train_loader:
                # print('imgs',imgs.shape,labels.shape)  #imgs torch.Size([128, 3, 224, 224]) torch.Size([128])
                imgs, label_train_f, label_train_c = imgs.to(self.device), label_train_f.to(
                    self.device), label_train_c.to(self.device)
                out_c, out_f = self.net(imgs)
                if my_lambda == 0:  # fine_train
                    loss_c = self.criterion_c(out_c, label_train_c).detach().to(self.device)
                    loss_f = self.criterion_f(out_f, label_train_f).to(self.device)
                    loss = loss_f
                elif my_lambda == 1:  # my_lambda =1时粗类损失, coarse_train
                    loss_c = self.criterion_c(out_c, label_train_c).to(self.device)
                    loss_f = self.criterion_f(out_f, label_train_f).detach().to(
                        self.device)  # detach_() Freezing back propagation
                    loss = loss_c
                else:
                    loss_c = self.criterion_c(out_c, label_train_c).to(self.device)
                    loss_f = self.criterion_f(out_f, label_train_f).to(self.device)
                    scale = self.get_scalingfac(loss_c, loss_f)
                    loss = my_lambda * loss_c + (1 - my_lambda) * scale * loss_f
                _, pred_c = torch.max(out_c, dim=1)
                _, pred_f = torch.max(out_f, dim=1)
                train_loss_epoch.append(loss.item())
                train_loss_epoch_c.append(loss_c.item())
                train_loss_epoch_f.append(loss_f.item())
                num_correct_c += torch.sum(pred_c == label_train_c)
                num_correct_f += torch.sum(pred_f == label_train_f)
                num_total += label_train_f.size(0)
                # ---------------------------------- compute Tie and FH-------------------------------------------------
                # coarse_num = self.args.num_classes_c
                # tie = util.EvaHier_TreeInducedError(self.tree, pred_f + 2 + coarse_num, label_train_f + 2 + coarse_num)
                # fh = util.EvaHier_HierarchicalPrecisionAndRecall(self.tree, pred_f + 2 + coarse_num,
                #                                                  label_train_f + 2 + coarse_num)
                # TIE.append(tie)
                # FH.append(fh)

                # #-------------------------------------------------------------------------------------------
                self.solver.zero_grad()
                loss.backward()
                self.solver.step()
            # train_tie_per_epoch = sum(TIE) / len(TIE)
            # train_fh_per_epoch = sum(FH) / len(FH)
            train_acc_epoch_c = num_correct_c.detach().cpu().numpy() * 100 / num_total
            train_acc_epoch_f = num_correct_f.detach().cpu().numpy() * 100 / num_total
            train_avg_loss_epoch_c = sum(train_loss_epoch_c) / len(train_loss_epoch_c)
            train_avg_loss_epoch_cf = sum(train_loss_epoch) / len(train_loss_epoch)
            train_avg_loss_epoch_f = sum(train_loss_epoch_f) / len(train_loss_epoch_f)
            test_loss_cf, test_loss_c, test_loss_f, test_acc_epoch_c, test_avg_epoch_tie, test_avg_epoch_fh, test_acc_epoch_f = self._accuracy_hier(
                epoch)
            test_acc.append(test_acc_epoch_f)
            train_acc.append(train_acc_epoch_f)
            if test_acc_epoch_f > best_acc:
                best_acc = test_acc_epoch_f  # best_acc更新
                best_epoch = epoch + 1
                print('*', end='')
                torch.save(self.net.state_dict(), self.path['path_model'] + '/' + 'Best' + '.pth')
            print(
                '{}\t\t {:0.4f}\t\t\t {:0.4f}\t\t\t\t{:0.4f}\t\t {:0.2f}%\t\t {:0.2f}%\t\t {:0.4f}\t\t {:0.4f}\t\t {:0.4f}\t\t {:0.2f}\t\t {:0.4f}\t\t {:0.2f}%\t\t {:0.2f}%'.format(
                    epoch + 1, train_avg_loss_epoch_cf, train_avg_loss_epoch_c, train_avg_loss_epoch_f,
                    train_acc_epoch_c,
                    train_acc_epoch_f, test_loss_cf, test_loss_c, test_loss_f, test_acc_epoch_c, test_avg_epoch_tie,
                    test_avg_epoch_fh, test_acc_epoch_f))

            f.write(
                '{}\t\t {:0.4f}\t\t\t {:0.4f}\t\t\t {:0.4f}\t\t {:0.2f}%\t\t {:0.2f}%\t\t {:0.4f}\t\t {:0.4f}\t\t {:0.4f}\t\t  {:0.2f}%\t\t {:0.4f}\t\t {:0.2f}%\t\t {:0.2f}%'.format(
                    epoch + 1, train_avg_loss_epoch_cf, train_avg_loss_epoch_c, train_avg_loss_epoch_f,
                    train_acc_epoch_c, train_acc_epoch_f, test_loss_cf, test_loss_c, test_loss_f, test_acc_epoch_c,
                    test_avg_epoch_tie, test_avg_epoch_fh,
                    test_acc_epoch_f))
            f.write("\n")
            f.flush()
        print('BestAcc: {:.2f}%, BestEP: {:.4f}'.format(best_acc, best_epoch))
        f.write('BestAcc: {:.2f}%, BestEP: {:.4f}'.format(best_acc, best_epoch))
        f.close()

        plt.figure()  # 画图
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')
        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title('Loss')
        plt.savefig(self.path['path_cur'] + '/' + 'Hier' + '.png')

    def train_flat(self):
        self.mkdir(self.path['path_cur'])
        self.mkdir(self.path['path_model'])
        f = open(self.path['path_log'], 'w')
        f.write('basic information parameter:' + str(self.args.flag) + '\t\t' + 'ratio:'
                + str(self.args.ratio) + '\t\t' + 'seed:' + str(self.args.seed) + '\t\t' + 'loss_type:'
                + str(self.args.loss_type) + '\t\t' + 'train_rule:' + str(self.args.train_rule) + '\n')
        f.write(
            'Epoch\t TrainLoss \t TrainAcc\t  TestLoss\t TIE\t FH\t TestAcc_F\n')
        epochs = np.arange(1, self.options['epochs'] + 1)
        print('Training process starts:...')
        print('Epoch\tTrainLoss\tTrainAcc\t Testloss\t Tie\t FH\t  TestAcc')
        print('_' * 76)
        best_acc = 0.0
        test_acc = list()
        train_acc = list()
        cls_num_list_c, cls_num_list_f = self.per_class_num()
        for epoch in range(self.options['epochs']):
            self.net.train(True)
            self.adjust_learning_rate(self.solver, epoch, self.args)
            train_loss_epoch_2 = list()
            num_correct = 0
            num_total = 0
            # TIE = list()
            # FH = list()

            # #----------------------------------strategy choice-------------------------------------------# #
            if epoch < self.options['warm']:
                self.train_loader = self.train_loader
                print('Ordinary 策略')
            else:
                self.train_loader = self.train_loader_oversample
                print('DRS策略')
            if self.args.train_rule == 'None':
                per_cls_weights = None
            elif self.args.train_rule == 'RW':
                per_cls_weights = 1 / np.array(cls_num_list_f)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            elif self.args.train_rule == 'DRW':
                if epoch < 160:
                    per_cls_weights = None
                else:
                    per_cls_weights = 1 / np.array(cls_num_list_f)
                    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                    per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            elif self.args.train_rule == 'CB':
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list_f)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            elif self.args.train_rule == 'DCB':
                idx = epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list_f)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_f)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

            # #----------------------------------Loss function choice--------------------------------------------# #
            if self.args.loss_type == 'CE':
                self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(self.device)
            elif self.args.loss_type == 'LDAM':
                self.criterion = losses.LDAMLoss(cls_num_list=cls_num_list_f, max_m=0.5, s=30,
                                                 weight=per_cls_weights).to(
                    self.device)
            elif self.args.loss_type == 'Focal':
                self.criterion = losses.FocalLoss(per_cls_weights, gamma=1).to(self.device)

            # #----------------------------------Train stage-----------------------------------------# #
            for imgs, label_train_f, label_train_c in self.train_loader:
                imgs, label_train_f, label_train_c = imgs.to(self.device), label_train_f.to(
                    self.device), label_train_c.to(self.device)
                output = self.net(imgs)
                loss = self.criterion(output, label_train_f)
                output = torch.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                num_correct += torch.sum(pred == label_train_f.detach_())
                train_loss_epoch_2.append(loss.item())
                num_total += label_train_f.size(0)
                self.solver.zero_grad()
                loss.backward()
                self.solver.step()
            train_acc_epoch = num_correct.detach().cpu().numpy() * 100 / num_total
            avg_loss_per_epoch = sum(train_loss_epoch_2) / len(train_loss_epoch_2)
            test_acc_epoch, test_loss_epoch_avg, tie, fh = self._accuracy_flat(epoch)
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            if test_acc_epoch > best_acc:
                best_acc = test_acc_epoch  # best_acc更新
                best_epoch = epoch + 1
                print('*', end='')
                torch.save(self.net.state_dict(),
                           self.path['path_model'] + '/' + 'best' + '.pth')
            print(
                '{}\t\t{:0.4f} \t\t{:0.2f}% \t\t{:0.4f}\t\t {:0.4f}\t{:0.2f}%\t{:0.2f}%'.format(epoch + 1,
                                                                                                avg_loss_per_epoch,
                                                                                                train_acc_epoch,
                                                                                                test_loss_epoch_avg,
                                                                                                tie, fh,
                                                                                                test_acc_epoch))
            f.write(
                '{}\t\t{:0.4f} \t\t{:0.2f}% \t\t{:0.4f}\t\t {:0.4f}\t\t {:0.2f}% \t\t{:0.2f}%'.format(epoch + 1,
                                                                                                      avg_loss_per_epoch,
                                                                                                      train_acc_epoch,
                                                                                                      test_loss_epoch_avg,
                                                                                                      tie, fh,
                                                                                                      test_acc_epoch))
            f.write("\n")
            f.flush()
        f.write('BestAcc: {:.2f}%, BestEP: {:.4f}'.format(best_acc, best_epoch))
        print('BestAcc: {:.2f}%, BestEP: {:.4f}'.format(best_acc, best_epoch))

    def _accuracy_hier(self, epoch):
        my_lambda = self.get_lambda(epoch, coarse_train_ep=self.args.coarse_ep, fine_train_ep=self.args.fine_ep)
        with torch.no_grad():
            self.net.eval()
            num_acc_c = 0
            num_acc_f = 0
            num_total = 0
            test_loss_c = []
            test_loss_f = []
            test_loss_cf = []
            TIE = []
            FH = []
            for imgs, label_test_f, label_test_c, in self.test_loader:
                imgs, label_test_f, label_test_c = imgs.to(self.device), label_test_f.to(self.device), label_test_c.to(
                    self.device)
                out_c, out_f = self.net(imgs)
                if my_lambda == 0:  # fine_train
                    loss_c = self.criterion_c(out_c, label_test_c).detach().to(self.device)
                    loss_f = self.criterion_f(out_f, label_test_f).to(self.device)
                    loss = loss_f
                elif my_lambda == 1:  # my_lambda =1时粗类损失, coarse_train
                    loss_c = self.criterion_c(out_c, label_test_c).to(self.device)
                    loss_f = self.criterion_f(out_f, label_test_f).detach().to(
                        self.device)  # detach_() Freezing back propagation
                    loss = loss_c
                else:
                    loss_c = self.criterion_c(out_c, label_test_c).to(self.device)
                    loss_f = self.criterion_f(out_f, label_test_f).to(self.device)
                    print(loss_c,loss_f)
                    scale = self.get_scalingfac(loss_c, loss_f)
                    loss = my_lambda * loss_c + (1 - my_lambda) * scale * loss_f
                _, pred_c = torch.max(out_c, dim=1)
                _, pred_f = torch.max(out_f, dim=1)
                test_loss_c.append(loss_c)
                test_loss_cf.append(loss)
                test_loss_f.append(loss_f)

                test_avg_epoch_lossC = sum(test_loss_c) / len(test_loss_c)
                test_avg_epoch_lossF = sum(test_loss_f) / len(test_loss_f)
                test_avg_epoch_lossCF = sum(test_loss_cf) / len(test_loss_cf)
                num_acc_c += torch.sum(pred_c == label_test_c)
                num_acc_f += torch.sum(pred_f == label_test_f)
                num_total += label_test_f.size(0)

                # -------------------------------------Compute TIE and FH-----------------------------------------
                if epoch > 160:
                    coarse_num = self.args.num_classes_c
                    tie = util.EvaHier_TreeInducedError(self.tree, pred_f + 2 + coarse_num,
                                                        label_test_f + 2 + coarse_num)
                    fh = util.EvaHier_HierarchicalPrecisionAndRecall(self.tree, pred_f + 2 + coarse_num,
                                                                     label_test_f + 2 + coarse_num)
                    TIE.append(tie)
                    FH.append(fh)
                    test_avg_epoch_tie = sum(TIE) / len(TIE)
                    test_avg_epoch_fh = sum(FH) / len(FH) * 100
                else:
                    test_avg_epoch_tie = 0
                    test_avg_epoch_fh = 0
                # ------------------------------------------------------------------------------------------------

        return test_avg_epoch_lossCF, test_avg_epoch_lossC, test_avg_epoch_lossF, \
               num_acc_c.detach().cpu().numpy() * 100 / num_total, \
               test_avg_epoch_tie, \
               test_avg_epoch_fh, \
               num_acc_f.detach().cpu().numpy() * 100 / num_total

    def _accuracy_flat(self, epoch):
        num_acc = 0
        num_total = 0
        self.net.eval()
        TIE = []
        FH = []
        test_loss_epoch = []
        with torch.no_grad():
            for imgs, label_test_f, label_test_c in self.test_loader:  # 1个test_loader有多个batch的16张图片，tensor([1,2,3,,....])，tensor([1,2,3,2,4,10...])
                imgs, label_test_f, label_test_c = imgs.to(self.device), label_test_f.to(self.device), label_test_c.to(
                    self.device)
                output = self.net(imgs)
                output = torch.softmax(output, dim=1)
                loss = self.criterion(output, label_test_f)
                _, pred = torch.max(output, dim=1)
                num_acc += torch.sum(pred == label_test_f.detach_())
                num_total += label_test_f.size(0)

                # -------------------------------------Compute TIE and FH-----------------------------------------
                if epoch > 160:
                    coarse_num = self.args.num_classes_c
                    tie = util.EvaHier_TreeInducedError(self.tree, pred + 2 + coarse_num,
                                                        label_test_f + 2 + coarse_num)
                    fh = util.EvaHier_HierarchicalPrecisionAndRecall(self.tree, pred + 2 + coarse_num,
                                                                     label_test_f + 2 + coarse_num)
                    TIE.append(tie)
                    FH.append(fh)
                    test_avg_epoch_tie = sum(TIE) / len(TIE)
                    test_avg_epoch_fh = sum(FH) / len(FH) * 100
                else:
                    test_avg_epoch_tie = 0
                    test_avg_epoch_fh = 0
                # ------------------------------------------------------------------------------------------------
                test_loss_epoch.append(loss.item())
                test_loss_epoch_avg = sum(test_loss_epoch) / len(test_loss_epoch)

        return num_acc.detach().cpu().numpy() * 100 / num_total, test_loss_epoch_avg, test_avg_epoch_tie, test_avg_epoch_fh
        # , test_avg_epoch_tie, test_avg_epoch_fh

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
                base_model = resnet32_hier.resnet32(self.args.num_classes_c, self.args.num_classes_f, use_norm=True)
            elif self.args.flag == 'smffresnet_hier':
                print("smffresnet_hier")
                base_model = smffresnet_hier.resnet32(self.args.num_classes_c, self.args.num_classes_f)
            print(base_model)
            print(base_model.linear_f)
        else:
            if self.args.flag == 'resnet_flat':
                base_model = resnet32_flat.resnet32(self.args.num_classes_f, use_norm=False)
                print('resnet_flat')
                print(base_model.linear)
            elif self.args.flag == 'smffresnet_flat':
                print("smffresnet_flat")
                base_model = smffresnet_flat.resnet32(self.args.num_classes_f)

        return base_model

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = args.base_lr * epoch / 5
        elif epoch > 160:
            lr = args.base_lr * 0.01
        elif epoch > 180:
            lr = args.base_lr * 0.0001
        else:
            lr = args.base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical classification experiments under long-tailed datasets'
    )
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--base_lr', type=float, default=0.1, help='base learning rate for training')
    parser.add_argument('--epochs', type=int, default=200, help='batch size for training')

    parser.add_argument('--coarse_ep', type=int, default=40, help='Control coarse training epoch for transferring')
    parser.add_argument('--fine_ep', type=int, default=90, help='Control fine training epoch for transferring')
    parser.add_argument('--num_classes_c', type=int, default=20, help='all unique coarse classes number')
    parser.add_argument('--num_classes_f', type=int, default=100, help='all unique fine classes number')

    parser.add_argument('--flag', type=str, default='resnet20_csms',
                        help='Hier_dataset is True = resnet_hier/resnetmff_hier/smffresnet_hier    false = resnet_flat/resnetmff_flat/smffresnet_flat/resnet20_orgain/resnet20_csms')
    # help='0：Hier_dataset is True =ResNet32_Hier else =ResNet32_Flat, 1:ResNet_MFF_Hier/ResNet_MFF_Flat')
    parser.add_argument('--Hier_dataset', type=bool, default=False, help='Whether is hierarchical structure')

    parser.add_argument('--Update_hier', type=bool, default=False, help='Whether to re-establish the tree structure')
    parser.add_argument('--ratio', type=int, default=200, help='unblance training')
    parser.add_argument('--dataset', type=int, default=100,
                        help='set dataset voc20 or cifar100 or tiredImagenet608 or Sun324 or iNaturealist5089')

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight_decay for SGD')
    parser.add_argument('--gpu_id', type=int, default=0, help='choose one gpu for training')
    parser.add_argument('--img_size', type=int, default=32, help='image\'s size for transforms')

    parser.add_argument('--warm', type=int, default=210, help='Controal oversampling, using flat state')
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--loss_type', type=str, default='CE',
                        help='Loss_type, coarse loss function and Fine Loss function, if is Hier=Two_CE, else CE ')
    parser.add_argument('--train_rule', type=str, default='None', help='train_type')
    parser.add_argument('--dataset_name', type=str, default='test1',
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
            train_root_data_path = 'cifar_100_long_tail_fen_blance'
        test_root_data_path = "public_Hier_test"
    elif args.dataset == 20:
        if args.ratio == 57:
            train_root_data_path = '/Voc20_datasets/VOC2012_Hier'
            test_root_data_path = '/Voc20_datasets/VOC2012_Hier'
    elif args.dataset == 324:
        if args.ratio == 28:
            train_root_data_path = '/SUN_datasets/Sun_hier'
            test_root_data_path = '/SUN_datasets/Sun_hier'
    elif args.dataset == 608:
        if args.ratio == 203:
            train_root_data_path = "/TiredImagenet_datasets/TiredImagnet_hier"
            test_root_data_path = "/TiredImagenet_datasets/TiredImagnet_hier"

    model_save_path = '/home/grczh/zwzw/TL_AT-master/Save_Base_Parmeters'
    path_cur = '/Result1/' + str(
        args.dataset_name) + '/' + str(args.ratio) + '/' + str(
        args.flag) + '_' + args.train_rule
    path_model = path_cur + '/model/'
    path_log = path_cur + '/result.txt'

    options = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'img_size': args.img_size,
        'dataset': args.dataset,
        'device': torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu'),
        'seed': args.seed,
        'warm': args.warm
    }
    path = {
        'model_save': model_save_path,
        'train_data_path': train_root_data_path,
        'test_data_path': test_root_data_path,
        'path_cur': path_cur,
        'path_model': path_model,
        'path_log': path_log
    }
    manager = NetworkManager(options, path, args)  
    if args.Hier_dataset:
        manager.train_hier()  # 开始训练
        manager.train_flat()


if __name__ == '__main__':
    main()
