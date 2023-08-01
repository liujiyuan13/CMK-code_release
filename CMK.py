import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
import math
import numpy as np
import random

import scipy.io as scio
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import tensorboard_logger as tb_logger
import argparse


def load_data(args):
    # load from mat
    data = scio.loadmat(os.path.join(args.data_dir, args.data_name + '.mat'))
    Xs = data['X'].squeeze().tolist()
    gt = data['gt'].squeeze()
    num_class = np.unique(gt).shape[0]
    feat_dims = []
    for i in range(len(Xs)):
        Xs[i] = torch.tensor(Xs[i].astype(np.float32).T, dtype=torch.float32).to(args.device)
        feat_dims.append(Xs[i].shape[1])

    return Xs, gt, num_class, feat_dims


def accuracy_score(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def purity_score(y_true, y_pred):
    """
    Calculate clustering purity.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi_score(y_true, y_pred):
    """
    Calculate clustering normalized mutual information (NMI).
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI, in [0,1]
    """

    # return NMI
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def cluster_metric(y_true, y_pred):
    """
    Calculate clustering metrics, including accuracy, NMI and purity.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        [accuracy, NMI, purity], in [0,1]
    """
    # compute accuracy, NMI and purity
    acc = accuracy_score(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)

    # return metrics
    return acc, nmi, pur


class FCNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, feat_dims=[100, 100], latent_dim=64, normalize=True):
        super(FCNet, self).__init__()
        self.feat_dims = feat_dims
        self.num_view = len(feat_dims)
        self.latent_dim = latent_dim
        self.normalize = normalize

        for i in range(self.num_view):
            exec('self.fc_{} = nn.Linear(self.feat_dims[{}], self.latent_dim, bias=False)'.format(i, i))

    def forward(self, x: list):
        tmp = []
        for i in range(self.num_view):
            if self.normalize:
                exec('tmp.append(F.normalize(self.fc_{}(x[{}]), dim=1))'.format(i, i))
            else:
                exec('tmp.append(self.fc_{}(x[{}]))'.format(i, i))
        return tmp


class ConLoss(nn.Module):

    def __init__(self, kernel_options, temperature=1.0, num_class=10, device=torch.device('cpu')):
        super(ConLoss, self).__init__()
        self.kernel_options = kernel_options
        self.temperature = temperature
        self.num_class = num_class
        self.device = device

    def forward(self, features):

        # flatten features
        num_view, num_smp = len(features), features[0].shape[0]
        features = torch.cat(features, dim=0)

        # get mask
        mask = torch.eye(num_smp, dtype=torch.float32).to(self.device)
        mask = mask.repeat(num_view, num_view)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_smp * num_view).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # define Euclidean distance
        def EuDist2(fea_a, fea_b):
            num_smp = fea_a.shape[0]
            aa = torch.sum(fea_a * fea_a, 1, keepdim=True)
            bb = torch.sum(fea_b * fea_b, 1, keepdim=True)
            ab = torch.matmul(fea_a, fea_b.T)
            D = aa.repeat([1, num_smp]) + bb.repeat([1, num_smp]) - 2 * ab
            return D

        # compute kernels
        if self.kernel_options['type'] == 'Gaussian':
            D = EuDist2(features, features)
            K = torch.exp(-D / (2 * self.kernel_options['t'] ** 2))
        elif self.kernel_options['type'] == 'Linear':
            K = torch.matmul(features, features.T)
        elif self.kernel_options['type'] == 'Polynomial':
            K = torch.pow(self.kernel_options['a'] * torch.matmul(features, features.T) + self.kernel_options['b'],
                          self.kernel_options['d'])
        elif self.kernel_options['type'] == 'Sigmoid':
            K = torch.tanh(self.kernel_options['d'] * torch.matmul(features, features.T) + self.kernel_options['c'])
        elif self.kernel_options['type'] == 'Cauchy':
            D = EuDist2(features, features)
            K = 1 / (D / self.kernel_options['sigma'] + 1)
        else:
            raise NotImplementedError

        # loss of contrastive learning
        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
        mean_log_prob_pos = - (mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        time1 = time.time()

        kernels = torch.zeros([num_smp, num_smp, num_view], dtype=torch.float32).to(self.device)
        for i in range(num_view):
            kernels[:, :, i] = K[i * num_smp: (i + 1) * num_smp, i * num_smp: (i + 1) * num_smp]
        kernel = kernels.mean(2)
        val, vec = torch.eig(kernel.detach(), eigenvectors=True)
        _, ind = torch.sort(val[:, 0], descending=True)
        H = vec[:, ind[:self.num_class]]

        # loss of extra downstream task
        loss_extra = (torch.trace(kernel) - torch.trace(torch.chain_matmul(H.T, kernel, H))) / num_smp

        # get normalized H for later validation
        H = F.normalize(H).detach().cpu().numpy()

        time_extra = time.time() - time1

        return loss_con, loss_extra, K.detach().cpu().numpy(), H, time_extra


def adjust_learning_rate(args, optimizer, epoch):

    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        # 3 segments: 1st (from 1 to 0) for cmk training, 2nd (from 0 to 1) and 3rd (from 1 to 0) for cmkkm training.
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch * 3 / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(Xs, model, criterion, optimizer, trade_off):
    """one epoch training"""
    # train mode
    model.train()

    # forward
    features = model(Xs)
    loss_con, loss_extra, K, H, time_extra = criterion(features)

    # overall loss
    loss = loss_con + trade_off * loss_extra

    # back propagation: SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    time1 = time.time()

    # output
    num_smp, num_view = Xs[0].shape[0], len(Xs)
    feas = [fea.detach().cpu().numpy() for fea in features]

    # get kernel
    kernels = np.zeros([num_smp, num_smp, num_view])
    for i in range(num_view):
        kernels[:, :, i] = K[i * num_smp: (i + 1) * num_smp, i * num_smp: (i + 1) * num_smp]
    del K

    # get k_diff
    k_diff, fea_diff = 0, 0
    for i in range(num_view):
        for j in range(num_view):
            if i != j:
                kernel_diff_ij = np.power(kernels[:, :, i] - kernels[:, :, j], 2)
                k_diff += 1 / (num_view ** 2 - num_view) * kernel_diff_ij.mean()
                fea_diff_ij = np.power(feas[i] - feas[j], 2)
                fea_diff += 1 / (num_view ** 2 - num_view) * fea_diff_ij.mean()

    time_extra_2 = time.time() - time1

    return loss_con.item(), loss_extra.item(), kernels, k_diff, fea_diff, H, time_extra, time_extra_2


def main(args):

    # set seed for reproduction
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # enable GPU
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    Xs, gt, num_class, feat_dims = load_data(args)

    # create model
    model = FCNet(feat_dims=feat_dims, latent_dim=args.latent_dim, normalize=args.normalize).to(device=args.device)
    criterion = ConLoss(args.kernel_options, args.temperature, num_class, device=args.device).to(device=args.device)

    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    time_running = time.time()
    lrs, losses_con, losses_extra, k_diffs, fea_diffs, accs, nmis, purs = [], [], [], [], [], [], [], []
    time_epochs = []
    for epoch in range(1, args.epochs + 1):
        time1 = time.time()

        # set learning rate
        adjust_learning_rate(args, optimizer, epoch)

        # 3 segments: 1st (from 1 to 0) for cmk training, 2nd (from 0 to 1) and 3rd (from 1 to 0) for cmkkm training.
        if epoch <= args.epochs // 3:
            trade_off = 0
        else:
            trade_off = args.trade_off

        # train for one epoch
        loss_con, loss_extra, kernels, k_diff, fea_diff, H, time_extra, time_extra_2 = train(Xs, model, criterion, optimizer, trade_off)

        time_epochs.append(time.time() - time1 - time_extra - time_extra_2)

        # calculate metrics
        kmeans = KMeans(n_clusters=num_class).fit(H)
        acc, nmi, pur = cluster_metric(gt, kmeans.labels_)
        accs.append(acc)
        nmis.append(nmi)
        purs.append(pur)

        # log
        lrs.append(optimizer.param_groups[0]['lr'])
        losses_con.append(loss_con)
        losses_extra.append(loss_extra)
        k_diffs.append(k_diff)
        fea_diffs.append(fea_diff)

        # print
        if epoch % args.print_freq == 0:
            print('  . epoch {}, time: {:.2f}, loss_con: {:.2f}, loss_extra: {:.2f}'.format(
                epoch, time.time() - time_running, loss_con, loss_extra))

        # save
        # regular save
        save_file = None
        if epoch == args.epochs // 3:
            save_file = os.path.join(args.save_path, '{}_cmk.mat'.format(args.kernel_options['type']))
        elif epoch == args.epochs*2 // 3:
            save_file = os.path.join(args.save_path, '{}_cmkkm_mid.mat'.format(args.kernel_options['type']))
        elif epoch == args.epochs:
            save_file = os.path.join(args.save_path, '{}_cmkkm.mat'.format(args.kernel_options['type']))
        # save by specified epochs
        save_file_2 = None
        if epoch in args.save_epochs:
            save_file_2 = os.path.join(args.save_path, '{}_epoch_{}.mat'.format(args.kernel_options['type'], epoch))
            
        # data to save
        save_dict = {'accs': accs, 'nmis': nmis, 'purs': purs, 'gt': gt, 'lrs': lrs,
                     'losses_con': losses_con, 'losses_extra': losses_extra, 'k_diffs': k_diffs,
                     'fea_diffs': fea_diffs, 'time_epochs': time_epochs}
        save_dict.update({'K': kernels})

        # operation to save
        if save_file is not None:
            scio.savemat(save_file, save_dict)
        if save_file_2 is not None:
            scio.savemat(save_file_2, save_dict)

    # clear GPU cache
    torch.cuda.empty_cache()


def default_args(data_name, normalize=True, latent_dim=128, learning_rate=1.0, epochs=300, save_epochs=[]):
    # params
    args = argparse.ArgumentParser().parse_args()

    # kernel
    args.kernel_options = dict()
    args.kernel_options['type'] = 'Gaussian'
    args.kernel_options['t'] = 1.0

    # net
    args.normalize = normalize  # [True, False]
    args.trade_off = 1  # the trade off between the CMK generation and MKC task
    args.latent_dim = latent_dim

    # net train
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    args.epochs = epochs  # the first epochs for CMK training, the last two epochs for CMKKM training (epochs%3 = 0)
    assert args.epochs % 3 == 0
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]  # not used here
    args.temperature = 1.0  # not used here

    # log and save
    args.print_freq = 100

    # path
    args.save_epochs = save_epochs
    args.data_dir = './data'
    args.save_dir = './save'
    args.data_name = data_name
    args.save_path = os.path.join(args.save_dir, args.data_name, 'norm_{}'.format(args.normalize),
                                  'dim_{}'.format(args.latent_dim), 'lr_{}'.format(args.learning_rate),
                                  'epochs_{}'.format(args.epochs))
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    return args


if __name__ == '__main__':
    data_name = 'BBC2'
    main(default_args(data_name))
