import torch
import torch.nn as nn
import torch.nn.functional as F


def LossFunction(loss_type):
    if loss_type == 'CrossEntropy':
        loss_fn = SampledCrossEntropyLoss
    elif loss_type == 'TOP1':
        loss_fn = TOP1Loss
    elif loss_type == 'BPR':
        loss_fn = BPRLoss
    elif loss_type == 'TOP1-max':
        loss_fn = TOP1_max
    elif loss_type == 'BPR-max':
        loss_fn = BPR_max
    else:
        raise NotImplementedError
    return loss_fn


xe_loss = nn.CrossEntropyLoss()
def SampledCrossEntropyLoss(logit):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    batch_size = logit.size(1)
    target = torch.arange(batch_size).long().to(logit.device)
    return xe_loss(logit, target)


def BPRLoss(logit):
    """
    Args:
        logit (BxB): Variable that stores the logits for the items in the session-parallel mini-batch.
                     Negative samples for a specific item are drawn from the other items in the
                     session-parallel minibatch, as mentioned in the original GRU4REC paper.
                     The first dimension corresponds to the batches, and the second dimension
                     corresponds to sampled number of items to evaluate.
    """
    # differences between the item scores
    diff = logit.diag().view(-1, 1).expand_as(logit) - logit
    # final loss
    loss = -torch.mean(F.logsigmoid(diff))

    return loss

def BPR_max(logit) :
    logit_softmax = F.softmax(logit, dim=1)
    diff = logit.diag().view(-1, 1).expand_as(logit) - logit
    loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
    return loss

    

    
def TOP1Loss(logit):
    """
    Args:
        logit (BxB): Variable that stores the logits for the items in the session-parallel mini-batch.
                     Negative samples for a specific item are drawn from the other items in the
                     session-parallel minibatch, as mentioned in the original GRU4REC paper.
                     The first dimension corresponds to the batches, and the second dimension
                     corresponds to sampled number of items to evaluate.
    """
    # differences between the item scores
    diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
    # final loss
    loss = F.sigmoid(diff).mean() + F.sigmoid(logit ** 2).mean()

    return loss

def TOP1_max(logit) :

    logit_softmax = F.softmax(logit, dim=1)
    diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
    loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
    return loss
