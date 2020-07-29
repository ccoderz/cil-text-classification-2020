#!/usr/bin/env python3

from src.utils.dataset import TweetsAsCharsAndWordsDataset
from src.utils.func import save_checkpoint, load_checkpoint, count_parameters
from src.models.CharWordCNN import CharAndWordCNN
import datetime
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import nn
from sklearn import metrics
import os
from tqdm import tqdm
tqdm.pandas()

ROOT_PATH = '/home/xiaochenzheng/DLProjects/cil-text-classification-2020'


class Obj:
    pass


args = Obj()
args.checkpoint_save_to_dir = os.path.join(ROOT_PATH, 'results', 'checkpoints')
args.checkpoint_continue_from = '' # os.path.join(ROOT_PATH, 'results', 'checkpoints', '...') --> just set this inline wherever most convenient
args.cuda = False
args.epochs = 10
args.max_norm = 1e3
args.val_interval = 2500 # 5000 # evaluate on the validation set every val_interval batch
args.max_samples = None # 100000 # TODO: set to None to use all samples
args.batch_size = 128
args.num_workers = 4
args.val_frac = 0.01 # 0.1 # reserve 0.1 = 10% of the training samples for validation.
# args.val_frac = None # Set to None to train on full training set, without validation.
args.if_change_lr = False
args.lr = 0.001

ALPHABET_SIZE = 70

ALPHABET_PATH = os.path.join(ROOT_PATH, 'aux-data', 'alphabet.json')
PREPROCESSED_TWITTER_DATASETS_DIR = os.path.join(ROOT_PATH, 'stanford_glove_preprocessed')
TWEETS_TRAIN_FILENAME = os.path.join(PREPROCESSED_TWITTER_DATASETS_DIR, 'dataset_stanfordglove_segmented_full.csv')
TWEETS_TEST_FILENAME = os.path.join(PREPROCESSED_TWITTER_DATASETS_DIR, 'test_stanfordglove_segmented.csv')


def eval(val_dataloader, model):
    criterion_reduc_sum = nn.BCEWithLogitsLoss(reduction='sum')  # sum of losses (instead of mean) over the batch
    if args.cuda:
        criterion_reduc_sum = criterion_reduc_sum.cuda()
    was_training = model.training  # don't forget to put it back in training mode at the end!
    model.eval()
    with torch.no_grad():
        predicates_all = []
        target_all = []
        accumulated_loss = 0
        tot_samples = 0
        for i_batch, data in enumerate(tqdm(val_dataloader)):
            inputs, target = data
            target = target.float()  # for some reason BCEWithLogitsLoss requires target to be float
            if args.cuda:
                #     inputs, target = inputs.cuda(), target.cuda()
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
                target = target.cuda()
            tot_samples += len(target)

            output_char, output_word, output = model(inputs)
            output_char = output_char.squeeze(1)  # (n, 1) -> (n,)
            output_word = output_word.squeeze(1)  # (n, 1) -> (n,)
            output = output.squeeze(1)  # (n, 1) -> (n,)
            assert output.shape == (target.shape[0],)  # train_dataloader.batch_size, except for the last batch
            loss = 0.001 * criterion_reduc_sum(output_char, target) + 0.001 * criterion_reduc_sum(output_word,
                                                                                                  target) + criterion_reduc_sum(
                output, target)

            accumulated_loss += loss.item()  # sum of losses (instead of mean) over the batch
            predicates = torch.round(torch.sigmoid(output))

            predicates_all.append(predicates)
            target_all.append(target)

            if args.cuda:
                torch.cuda.synchronize()
    if was_training:
        model.train()

    avg_loss = accumulated_loss / tot_samples
    predicates_all = torch.cat(predicates_all).cpu()
    target_all = torch.cat(target_all).cpu()
    accuracy = metrics.accuracy_score(target_all, predicates_all)
    f1_score = metrics.f1_score(target_all, predicates_all)
    print(f'Validation - \
        \n\t loss: {accumulated_loss / tot_samples}  \
        \n\t acc: {accuracy} \
        \n\t f1-score: {f1_score} \
    ')
    # if args.log_result:
    #     with open(os.path.join(path, args.save_folder,'result_res.csv'), 'a') as r:
    #         r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
    #                                                         batch_train,
    #                                                         avg_loss,
    #                                                         accuracy,
    #                                                         optimizer.state_dict()['param_groups'][0]['lr']))
    return avg_loss, accuracy


def predict(test_dataloader, model):
    assert not test_dataloader.dataset.is_labeled  # the samples we get from test_dataloader are inputs only, no labels!
    was_training = model.training  # don't forget to put it back in training mode at the end!
    model.eval()
    with torch.no_grad():
        y_pred = []
        for i_batch, data in enumerate(tqdm(test_dataloader)):
            inputs = data
            # inputs = inputs[::-1] # TODO: check that it's in the right order
            if args.cuda:
                #     inputs = inputs.cuda()
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()

            _, _, logit = model(inputs)
            logit = logit.squeeze(1)
            assert logit.shape == (inputs[0].shape[0],)  # test_dataloader.batch_size, except for the last batch

            predicates = torch.round(torch.sigmoid(logit))
            y_pred.append(predicates)

            if args.cuda:
                torch.cuda.synchronize()
    if was_training:
        model.train()

    y_pred = torch.cat(y_pred)
    return y_pred


def main():

    # train_dataset = TweetsAsCharsAndWordsDataset(TWEETS_TRAIN_FILENAME, ALPHABET_PATH, is_labeled=True)
    train_dataset = TweetsAsCharsAndWordsDataset(TWEETS_TRAIN_FILENAME, ALPHABET_PATH, is_labeled=True,
                                                 max_samples=args.max_samples, vector_cache_path=os.path.join(ROOT_PATH, 'aux-data'))
    assert train_dataset.raw_nb_feats == ALPHABET_SIZE

    if args.val_frac:
        val_size = int(args.val_frac * len(train_dataset))
        train_size = len(train_dataset) - val_size

        torch.manual_seed(0)
        # need random_split to be deterministic if we want to avoid information leak when we reload notebook in-between training epochs
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        torch.manual_seed(torch.initial_seed())

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    # len(train_dataloader), len(val_dataloader) if args.val_frac else None  # number of batches. multiply by args.batch_size to get (approximate) number of samples.
    if args.val_frac:
        print('There are {} training samples and {} validation samples'.format(len(train_dataloader), len(val_dataloader)))
    else:
        None

    model = CharAndWordCNN()
    model_nickname = 'CharAndWordCNN'
    print(f'{count_parameters(model)} parameters')

    criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        # model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters())  # TODO: try tweaking parameters (e.g learning rate)

    for param_group in optimizer.param_groups:
        # param_group['lr'] = 0.0001
        print('learning rate is {}'.format(param_group['lr']))

    if args.checkpoint_continue_from:
        print(f'=> loading checkpoint from {args.checkpoint_continue_from}')
        checkpoint = load_checkpoint(model, optimizer,
                                     args.checkpoint_continue_from, args)  # load the state to `model` and `optimizer` and fetch the remaining info into `checkpoint`

        # always assume that we saved a model after an epoch finished, so start at the next epoch.
        start_epoch = checkpoint['epoch'] + 1
        # load optimizer, default all parameters are in cpu     --> pretty sure it's always a noop, but just in case
        if args.cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    else:
        start_epoch = 1

    print('Training will start at epoch {}'.format(start_epoch))

    for param_group in optimizer.param_groups:
        print('former learning rate is {}'.format(param_group['lr']))
        if args.if_change_lr:
            param_group['lr'] = args.lr
        print('now, learning rate is {}'.format(param_group['lr']))

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n\n===== Starting epoch #{epoch} =====')
        accumulated_train_loss = 0
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            inputs, target = data
            target = target.float()  # for some reason BCEWithLogitsLoss requires target to be float
            if args.cuda:
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output_char, output_word, output = model(inputs)
            output_char = output_char.squeeze(1)  # (n, 1) -> (n,)
            output_word = output_word.squeeze(1)  # (n, 1) -> (n,)
            output = output.squeeze(1)  # (n, 1) -> (n,)
            assert output.shape == (target.shape[0],)  # train_dataloader.batch_size, except for the last batch
            loss = 0.001 * criterion(output_char, target) + 0.001 * criterion(output_word, target) + criterion(output,
                                                                                                               target)
            accumulated_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            if args.cuda:
                torch.cuda.synchronize()

            # if args.verbose:
            #     print('\nTargets, Predicates')
            #     print(torch.cat((target.unsqueeze(1), torch.unsqueeze(torch.max(logit, 1)[1].view(target.size()).data, 1)), 1))
            #     print('\nLogit')
            #     print(logit)
            # if i_batch % args.log_interval == 0:
            #     corrects = (torch.round(torch.sigmoid(logit)) == target.data).float().sum()  # convert into float for division
            #     accuracy = 100.0 * corrects/args.batch_size
            #     print('Epoch[{}] Batch[{}] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{})'.format(epoch,
            #                                                                   i_batch,
            #                                                                   loss.data,
            #                                                                   optimizer.state_dict()['param_groups'][0]['lr'],
            #                                                                   accuracy,
            #                                                                   corrects,
            #                                                                   args.batch_size))
            if (i_batch + 1) % args.val_interval == 0:
                print(f'Training - loss: {accumulated_train_loss / (i_batch + 1)}')
                val_loss, val_acc = eval(val_dataloader, model)

        print(f'----- Finished epoch #{epoch} -----')
        # validation
        print('\nTraining - loss: {:.6f}'.format(accumulated_train_loss / i_batch))
        val_loss, val_acc = eval(val_dataloader, model)

        # save the model as this epoch
        if args.checkpoint_save_to_dir:
            ts = datetime.datetime.now().isoformat()
            file_path = os.path.join(args.checkpoint_save_to_dir, f'{model_nickname}_epoch_{epoch}_{ts}.pth.tar')
            print(f'=> saving checkpoint model to {file_path}')
            save_checkpoint(model,
                            optimizer,
                            {'epoch': epoch,
                             'validation_accuracy': val_acc},
                            file_path)

        start_epoch = epoch + 1

    print(f'finished the required number of epochs args.epoch={args.epoch}')


def pred():
    args.checkpoint_continue_from = os.path.join(ROOT_PATH, 'results', 'checkpoints',
                                                 'CharAndWordCNNv3_epoch_4_2020-07-28T21_08_45.530055.pth.tar')
    test_dataset = TweetsAsCharsAndWordsDataset(TWEETS_TEST_FILENAME, ALPHABET_PATH, is_labeled=False, vector_cache_path=os.path.join(ROOT_PATH, 'aux-data'))
    assert test_dataset.raw_nb_feats == ALPHABET_SIZE
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,  # need to keep the ordering of the tweets
                                 num_workers=0)
    len(test_dataloader)

    model = CharAndWordCNN()
    model_nickname = 'CharAndWordCNN'
    print(f'{count_parameters(model)} parameters')

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        # model = model.cuda()

    optimizer = optim.Adam(model.parameters())  # TODO: try tweaking parameters (e.g learning rate)

    print(f'=> loading checkpoint from {args.checkpoint_continue_from}')
    checkpoint = load_checkpoint(model, optimizer,
                                 args.checkpoint_continue_from, args)  # load the state to `model` and `optimizer` and fetch the remaining info into `checkpoint`

    y_pred = predict(test_dataloader, model)

    ts = datetime.datetime.now().isoformat()
    SUBMISSION_FILENAME = os.path.join(ROOT_PATH, f'{model_nickname}_submission_{ts}.csv')

    with open(SUBMISSION_FILENAME, 'w') as f:
        f.write('Id,Prediction\n')
        for i, label in enumerate(y_pred, start=1):
            f.write(f'{i},{label}\n')

    print(f'wrote to {SUBMISSION_FILENAME}')


if __name__ == '__main__':
    # pred()
    main()