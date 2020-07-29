#!/usr/bin/env python3
from termcolor import cprint, colored as c
import torch
import os


def inc(d, label):
    if label in d:
        d[label] += 1
    else:
        d[label] = 1


def precision_recall(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    labels = set(target)
    TP = {}
    TP_plus_FN = {}
    TP_plus_FP = {}
    for i in range(len(output)):

        inc(TP_plus_FN, target[i])
        inc(TP_plus_FP, output[i])
        if target[i] == output[i]:
            inc(TP, output[i])

    for label in labels:
        if label not in TP_plus_FN:
            TP_plus_FN[label] = 0
        if label not in TP_plus_FP:
            TP_plus_FP[label] = 0

    precision = {label: 0. if TP_plus_FP[label] ==0 else ((TP[label] if label in TP else 0) / float(TP_plus_FP[label])) for label in labels}
    recall = {label: 0. if TP_plus_FN[label] ==0 else ((TP[label] if label in TP else 0) / float(TP_plus_FN[label])) for label in labels}

    return precision, recall, TP, TP_plus_FN, TP_plus_FP


def F_score(p, r):

    f_scores = {
        label: None if p[label] == 0 and r[label] == 0 else (0 if p[label] == 0 or r[label] == 0 else 2 / (1 / p[label] + 1 / r[label]))
        for label in p
    }
    return f_scores


def print_f_score(output, target):
    """returns:
        p<recision>,
        r<ecall>,
        f<-score>,
        {"TP", "p", "TP_plus_FP"} """
    p, r, TP, TP_plus_FN, TP_plus_FP = precision_recall(output, target)
    f = F_score(p, r)

    for label in f.keys():
        cprint("Label: " + c(("  " + str(label))[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[label] * 100)[-5:], 'green') + '%' +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FP[label]).ljust(14) +
               "Recall: " + c("  {:.1f}".format((r[label] if label in r else 0) * 100)[-5:], 'green') + "%" +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FN[label]).ljust(14) +
               "F-Score: " + ("  N/A" if f[label] is None else (c("  {:.1f}".format(f[label] * 100)[-5:], "green") + "%"))
               )
    # return p, r, f, _


def save_checkpoint(model, optimizer, checkpoint, filename):
    """
    Args:
        optimizer: can be set to None, then no optimizer will be saved
        checkpoint is a dict that can be prepopulated (e.g with keys 'epoch' and 'validation_accuracy')
    """
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    if isinstance(model, torch.nn.DataParallel):
        assert not isinstance(model.module, torch.nn.DataParallel)  # check we didn't wrap multiple times by mistake...
        checkpoint['model_state_dict'] = model.module.state_dict()
    else:
        checkpoint['model_state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename, args):
    """
    Args:
        optimizer: can be set to None, then the optimizer state will be ignored (if there is one stored in the checkpoint)
            MUST be set to None if no optimizer state is stored in the checkpoint (so as to minimize risks of confusion)
    """
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    assert os.path.isfile(filename), f"no checkpoint found at {filename} (no such file)"
    # try to make it so that cpu->cpu, gpu->gpu, cpu->gpu, gpu->cpu all work (not 100% sure but I think this should do it)
    if args.cuda:
        device = torch.device("cuda")
        checkpoint = torch.load(filename, map_location=device)
    else:
        device = torch.device("cpu")
        checkpoint = torch.load(filename, map_location=device)
    # checkpoint = torch.load(filename) # or just don't worry abt it and pray that it works

    if isinstance(model, torch.nn.DataParallel):
        assert not isinstance(model.module, torch.nn.DataParallel)  # check we didn't wrap multiple times by mistake...
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model = model.cuda()  # possibly always a noop but just in case

    loaded_optimizer = False
    if 'optimizer_state_dict' in checkpoint.keys():
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded_optimizer = True
    else:
        assert optimizer is None, "Argument `optimizer` MUST be set to None if no optimizer state is stored in the checkpoint"

    if loaded_optimizer:
        print("successfully loaded model and optimizer states from checkpoint (in place)")
    else:
        print("successfully loaded model state from checkpoint (in place). (Did NOT load optimizer state.)")

    if 'epoch' in checkpoint:
        print(f"the model was trained for {checkpoint['epoch']} epochs")
    if 'validation_accuracy' in checkpoint:
        print(f"the model had achieved validation accuracy {checkpoint['validation_accuracy']}")
    return checkpoint


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

