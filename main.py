#!/usr/bin/env python3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch import nn
import argparse
import datetime
import errno
import torch
import json
import sys
import os
from src.utils.dataset import TweetsDataset
from src.utils.func import print_f_score
from src.models import CharResCNN_GRU, CharCNN, BERTGRU
path = '/home/xiaochenzheng/Desktop/CIL'

parser = argparse.ArgumentParser(description='Character level CNN text classifier training')
# data
parser.add_argument('--train_path', metavar='DIR',
                    help='path to training data csv [default: data/ag_news_csv/train.csv]',
                    default=os.path.join(path, 'train_full.csv'))
parser.add_argument('--val_path', metavar='DIR',
                    help='path to validation data csv [default: data/ag_news_csv/test.csv]',
                    default=os.path.join(path, 'val_full.csv'))
# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=200, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')
learn.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='Adam', help='Type of optimizer. SGD|Adam|ASGD are supported [default: Adam]')
learn.add_argument('--class_weight', default=None, action='store_true', help='Weights should be a 1D Tensor assigning weight to each of the classes.')
learn.add_argument('--dynamic_lr', action='store_true', default=True, help='Use dynamic learning schedule.')
learn.add_argument('--milestones', nargs='+', type=int, default=[5,10,15], help=' List of epoch indices. Must be increasing. Default:[5,10,15]')
learn.add_argument('--decay_factor', default=0.5, type=float, help='Decay factor for reducing learning rate [default: 0.5]')
# model (text classifier)
cnn = parser.add_argument_group('Model options')
cnn.add_argument('--alphabet_path', default=os.path.join(path, 'alphabet.json'), help='Contains all characters for prediction')
cnn.add_argument('--l0', type=int, default=501, help='maximum length of input sequence to CNNs [default: 1014]')
cnn.add_argument('--shuffle', action='store_true', default=True, help='shuffle the data every epoch')
cnn.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
cnn.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel')
cnn.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=1, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False, help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model, we can use /content/drive/My Drive/cil/models_CharResCNN/CharResCNN_best.pth.tar')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true', help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint_per_batch', default=10000, type=int, help='Save checkpoint per batch. 0 means never save [default: 10000]')
experiment.add_argument('--save_folder', default='models_CharResCNN_gru_tiny', help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--val_interval', type=int, default=10000, help='how many steps to wait before vaidation [default: 200]')
experiment.add_argument('--save_interval', type=int, default=1, help='how many epochs to wait before saving [default:1]')


def train(train_loader, dev_loader, model, args):
    # optimization scheme
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr)

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        # checkpoint = torch.load(args.continue_from, map_location=torch.device('cpu'))
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_acc = checkpoint.get('best_acc', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load optimizer, default all parameters are in cpu
        if args.cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    else:
        start_epoch = 1
        start_iter = 1
        best_acc = None

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 0.00005
    # dynamic learning scheme
    if args.dynamic_lr and args.optimizer != 'Adam':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.decay_factor,
                                                   last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=1e-3)
    # multi-gpu
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        # model = model.cuda()
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        criterion = criterion.cuda()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()  # not necessary
        if args.dynamic_lr and args.optimizer != 'Adam':
            scheduler.step()

        accumulated_train_loss = 0
        for i_batch, data in enumerate(train_loader, start=start_iter):
            inputs, target = data

            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            # inputs = Variable(inputs)
            # target = Variable(target)
            logit = model(inputs).squeeze(1)
            # loss = F.nll_loss(logit, target)
            optimizer.zero_grad()
            loss = criterion(logit, target)
            accumulated_train_loss += criterion(logit, target).item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            if args.cuda:
                torch.cuda.synchronize()

            if args.verbose:
                print('\nTargets, Predicates')
                print(torch.cat(
                    (target.unsqueeze(1), torch.unsqueeze(torch.max(logit, 1)[1].view(target.size()).data, 1)), 1))
                print('\nLogit')
                print(logit)

            if i_batch % args.log_interval == 0:
                corrects = (torch.round(
                    torch.sigmoid(logit)) == target.data).float().sum()  # convert into float for division
                accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{})'.format(epoch,
                                                                                                    i_batch,
                                                                                                    loss.data,
                                                                                                    optimizer.state_dict()[
                                                                                                        'param_groups'][
                                                                                                        0]['lr'],
                                                                                                    accuracy,
                                                                                                    corrects,
                                                                                                    args.batch_size))
            if i_batch % args.val_interval == 0:
                val_loss, val_acc = eval(dev_loader, model, epoch, i_batch, optimizer, args)
                print('\nTraining - loss: {:.6f}'.format(accumulated_train_loss / i_batch))

            # i_batch += 1 # output the number of batches to validation
        if args.checkpoint and epoch % args.save_interval == 0:
            # TODO
            file_path = os.path.join(path, '%s/CharResCNN_epoch_%d.pth.tar' % (args.save_folder, epoch))
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optimizer.state_dict(),
                                    'best_acc': best_acc},
                            file_path)

        # validation
        val_loss, val_acc = eval(dev_loader, model, epoch, i_batch, optimizer, args)
        print('\nTraining - loss: {:.6f}'.format(accumulated_train_loss / i_batch))
        # save best validation epoch model
        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            file_path = os.path.join(path, '%s/CharResCNN_best.pth.tar' % (args.save_folder))
            print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optimizer.state_dict(),
                             'best_acc': best_acc},
                            file_path)
        print('\n')


def eval(data_loader, model, epoch_train, batch_train, optimizer, args):
    criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        criterion = criterion.cuda()
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    with torch.no_grad():

        for i_batch, data in enumerate(data_loader):
            inputs, target = data
            # target.sub_(1)

            size += len(target)
            batch_num = i_batch + 1
            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            # inputs = Variable(inputs, volatile=True)
            # target = Variable(target)
            logit = model(inputs).squeeze(1)
            predicates = torch.round(torch.sigmoid(logit))
            accumulated_loss += criterion(logit, target).item()
            corrects += (torch.round(torch.sigmoid(logit)) == target.data).float().sum()
            predicates_all += predicates.cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()
            if args.cuda:
                torch.cuda.synchronize()

    avg_loss = accumulated_loss / batch_num
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{}) '.format(avg_loss,
                                                                                  optimizer.state_dict()[
                                                                                      'param_groups'][0]['lr'],
                                                                                  accuracy,
                                                                                  corrects,
                                                                                  size))
    print_f_score(predicates_all, target_all)
    print('\n')
    if args.log_result:
        with open(os.path.join(path, args.save_folder, 'result_res.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            accuracy,
                                                            optimizer.state_dict()['param_groups'][0]['lr']))

    return avg_loss, accuracy


def save_checkpoint(model, state, filename):
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)


def make_data_loader(dataset_path, alphabet_path, l0, batch_size, num_workers):
    print("\nLoading data from {}".format(dataset_path))
    dataset = TweetsDataset(label_data_path=dataset_path, alphabet_path=alphabet_path, l0=l0)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    return dataset, dataset_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # parse arguments
    args = parser.parse_args(args=[])

    # load train and dev data
    train_dataset, train_loader = make_data_loader(args.train_path,
                                                   args.alphabet_path, args.l0, args.batch_size, args.num_workers)
    dev_dataset, dev_loader = make_data_loader(args.val_path,
                                               args.alphabet_path, args.l0, args.batch_size, args.num_workers)

    # feature length
    args.num_features = len(train_dataset.alphabet)

    # get class weights
    class_weight, num_class_train = train_dataset.getClassWeight()
    _, num_class_dev = dev_dataset.getClassWeight()

    # when you have an unbalanced training set
    if args.class_weight != None:
        args.class_weight = torch.FloatTensor(class_weight).sqrt_()
        if args.cuda:
            args.class_weight = args.class_weight.cuda()

    print('\nNumber of training samples: {}'.format(str(train_dataset.__len__())))
    for i, c in enumerate(num_class_train):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))
    print('\nNumber of developing samples: {}'.format(str(dev_dataset.__len__())))
    for i, c in enumerate(num_class_dev):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    # make save folder
    try:
        os.makedirs(os.path.join(path, args.save_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    # args.save_folder = os.path.join(args.save_folder, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(path, args.save_folder, 'result_res.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))
    # model
    model = CharResCNN_GRU(args)
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("\nThe trainable parameters are parameters of the GRU (rnn) and the linear layer (out):"+"-"*60)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print("-"*60)

    # train
    train(train_loader, dev_loader, model, args)


if __name__ == "__main__":
    main()
