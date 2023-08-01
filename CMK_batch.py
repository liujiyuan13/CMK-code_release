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
import scipy.sparse as scsp
import h5py as hp
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import tensorboard_logger as tb_logger
import argparse



def read_mymat73(data_path, sparse=False):

    f = hp.File(data_path, 'r')

    Y = f['Y'][()].T.astype(np.int32)

    Xr = f['X'][()]
    Xr = Xr.reshape((Xr.shape[1],))
    X = []
    if sparse:
        for x in Xr:
            data = f[x]['data'][()]
            ir = f[x]['ir'][()]
            jc = f[x]['jc'][()]
            X.append(scsp.csc_matrix((data, ir, jc)).toarray())
    else:
        for x in Xr:
            X.append(f[x][()].T.astype(np.float64))

    return X, Y



def load_data(args):

    data_path = os.path.join(args.data_dir, args.data_name + '.mat')
    try:
        data = scio.loadmat(data_path)
        Xs = data['X'].squeeze().tolist()
        gt = data['Y'].squeeze()
    except:
        try:
            Xs, gt = read_mymat73(data_path)
        except:
            Xs, gt = read_mymat73(data_path, sparse=True)

    num_class = np.unique(gt).shape[0]
    feat_dims = []
    for i in range(len(Xs)):
        Xs[i] = Xs[i].astype(np.float32).T
        feat_dims.append(Xs[i].shape[1])

    return Xs, gt, num_class, feat_dims


class AverageMeter(object):
    '''
    compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
        tmp, weights = [], []
        for i in range(self.num_view):
            if self.normalize:
                exec('tmp.append(F.normalize(self.fc_{}(x[{}]), dim=1))'.format(i, i))
            else:
                exec('tmp.append(self.fc_{}(x[{}]))'.format(i, i))
            exec('weights.append(self.fc_{}.weight)'.format(i))
        return tmp, weights


class ConLoss(nn.Module):

    def __init__(self, kernel_options, temperature=1.0, num_class=10, device=torch.device('cpu')):
        super(ConLoss, self).__init__()
        self.kernel_options = kernel_options
        self.temperature = temperature
        self.num_class = num_class
        self.device = device


    def forward(self, features, trade_off):

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

        if trade_off != 0:
            kernels = torch.zeros([num_smp, num_smp, num_view], dtype=torch.float32).to(self.device)
            for i in range(num_view):
                kernels[:, :, i] = K[i * num_smp: (i + 1) * num_smp, i * num_smp: (i + 1) * num_smp]
            kernel = kernels.mean(2)
            val, vec = np.linalg.eig(kernel.detach().cpu().numpy())
            val, vec = val.real, vec.real # convert real number by removing the imaginary part
            ind = np.argsort(val)[::-1]
            H_out = vec[:, ind[:self.num_class]]
            H = torch.from_numpy(H_out).to(self.device)

            # val, vec = torch.eig(kernel.detach(), eigenvectors=True) # Not use this, will cause memory leakage
            # _, ind = torch.sort(val[:, 0], descending=True)
            # H = vec[:, ind[:self.num_class]]

            # loss of extra downstream task
            loss_extra = (torch.trace(kernel) - torch.trace(torch.chain_matmul(H.T, kernel, H))) / num_smp

            # get normalized H for later validation
            H_out = H_out / np.expand_dims(np.linalg.norm(H_out, axis=1), axis=1)

        else:
            loss_extra, H_out = torch.zeros(1, device=self.device), None

        time_extra = time.time() - time1

        return loss_con, loss_extra, K.detach().cpu().numpy(), H_out, time_extra


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

    return lr


def train(Xs, model, criterion, optimizer, trade_off):
    """one epoch training"""

    # train mode
    model.train()

    # forward
    features, weights = model(Xs)
    loss_con, loss_extra, K, H, time_extra = criterion(features, trade_off)

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

    # # get kernels
    kernels = np.zeros([num_smp, num_smp, num_view])
    for i in range(num_view):
        kernels[:, :, i] = K[i * num_smp: (i + 1) * num_smp, i * num_smp: (i + 1) * num_smp]

    # get k_diff
    k_diff, fea_diff = 0, 0
    for i in range(num_view):
        for j in range(num_view):
            if i != j:
                k_diff += 1 / (num_view ** 2 - num_view) * np.power(kernels[:, :, i] - kernels[:, :, j], 2).mean()
                fea_diff += 1 / (num_view ** 2 - num_view) * np.power(feas[i] - feas[j], 2).mean()

    # get weights
    weights = [weight.detach().cpu().numpy() for weight in weights]

    time_extra_2 = time.time() - time1

    return loss_con.item(), loss_extra.item(), k_diff, fea_diff, H, weights, time_extra, time_extra_2

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

        # set learning rate
        lr = adjust_learning_rate(args, optimizer, epoch)

        # 3 segments: 1st (from 1 to 0) for cmk training, 2nd (from 0 to 1) and 3rd (from 1 to 0) for cmkkm training.
        if epoch <= args.epochs // 3:
            trade_off = 0
        else:
            trade_off = args.trade_off
            return

        # separate in batch
        num_data = Xs[0].shape[0]
        rand_ind = np.random.permutation(num_data)
        loss_con, loss_extra, k_diff, fea_diff = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        acc, nmi, pur = AverageMeter(), AverageMeter(), AverageMeter()
        time_batch = 0
        for b in range(num_data//args.batch_size):
            time1 =  time.time()
            # get batch
            batch_ind = rand_ind[b*args.batch_size:(b+1)*args.batch_size]
            Xs_batch = []
            for v in range(len(Xs)):
                Xs_batch.append(torch.tensor(Xs[v][batch_ind,:]).to(args.device))

            # train for one epoch
            loss_con_tmp, loss_extra_tmp, k_diff_tmp, fea_diff_tmp, H, weights, time_extra, time_extra_2 = train(Xs_batch, model, criterion, optimizer, trade_off)

            time_batch += (time.time() - time1 - time_extra - time_extra_2)

            loss_con.update(loss_con_tmp, args.batch_size)
            loss_extra.update(loss_extra_tmp, args.batch_size)
            k_diff.update(k_diff_tmp, args.batch_size)
            fea_diff.update(fea_diff_tmp, args.batch_size)

            if H is not None:
                # calculate metrics
                kmeans = KMeans(n_clusters=num_class).fit(H)
                acc_tmp, nmi_tmp, pur_tmp = cluster_metric(gt[batch_ind], kmeans.labels_)
                acc.update(acc_tmp, args.batch_size)
                nmi.update(nmi_tmp, args.batch_size)
                pur.update(pur_tmp, args.batch_size)

        # log
        lrs.append(lr)
        losses_con.append(loss_con.avg)
        losses_extra.append(loss_extra.avg)
        k_diffs.append(k_diff.avg)
        fea_diffs.append(fea_diff.avg)
        accs.append(acc.avg)
        nmis.append(nmi.avg)
        purs.append(pur.avg)

        time_epochs.append(time_batch)

        # print
        if epoch % args.print_freq == 0:
            print('  . epoch {}, time: {:.2f}, loss_con: {:.2f}, loss_extra: {:.2f}, acc: {:.4f}'.format(
                epoch, time.time() - time_running, loss_con.avg, loss_extra.avg, acc.avg))

        # save
        save_file = None
        if epoch == args.epochs // 3:
            save_file = os.path.join(args.save_path, '{}_cmk.mat'.format(args.kernel_options['type']))
        elif epoch == args.epochs*2 // 3:
            save_file = os.path.join(args.save_path, '{}_cmkkm_mid.mat'.format(args.kernel_options['type']))
        elif epoch == args.epochs:
            save_file = os.path.join(args.save_path, '{}_cmkkm.mat'.format(args.kernel_options['type']))
        if save_file is not None:
            save_dict = {'accs': accs, 'nmis': nmis, 'purs': purs, 'gt': gt, 'lrs': lrs,
                         'losses_con': losses_con, 'losses_extra': losses_extra, 'k_diffs': k_diffs,
                         'fea_diffs': fea_diffs, 'time_epochs': time_epochs}
            weights_save = np.empty((len(weights),), dtype=np.object)
            for iw in range(len(weights)):
                weights_save[iw] = weights[iw]
            save_dict.update({'weights': weights_save})
            scio.savemat(save_file, save_dict)

    # clear GPU cache
    torch.cuda.empty_cache()


def default_args(data_name, normalize, latent_dim, batch_size=2048, learning_rate=1.0, epochs=90):
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
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    # args.epochs = 300  # the first 100 epochs for CMK training, the last 200 epochs for CMKKM training (epochs%3 = 0)
    args.epochs = epochs  # the first 100 epochs for CMK training, the last 200 epochs for CMKKM training (epochs%3 = 0)
    assert args.epochs % 3 == 0
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]  # not used here
    args.temperature = 1.0  # not used here

    # log and save
    args.print_freq = 10

    # path
    args.data_dir = './data'
    args.save_dir = './save_batch_R1'
    args.data_name = data_name
    args.save_path = os.path.join(args.save_dir, args.data_name, 'norm_{}'.format(args.normalize),
                                  'dim_{}'.format(args.latent_dim), 'batch_size_{}'.format(args.batch_size),
                                  'lr_{}'.format(args.learning_rate), 'epochs_{}'.format(args.epochs))
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    return args


if __name__ == '__main__':
    data_name = 'bbcsport_2view'
    main(default_args(data_name))
