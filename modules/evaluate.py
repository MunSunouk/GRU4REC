import torch

def get_recall(indices, targets):
    """ Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = n_hits / targets.size(0)
    
    return recall

def get_precision(indices, targets) :
    """ Calculates the precision score for the given predictions and targets
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        precision (float): the precision score
    """    
    targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    precision = n_hits / indices.size(-1)
    return precision

def get_f1score(recall,precision) :

    f1score = 2 * (precision * recall / (precision + recall))
    return f1score

def get_mrr(indices, targets):
    """ Calculates the MRR score for the given predictions and targets
    
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1,1).expand_as(indices)
    # ranks of the targets, if it appears in your indices
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(rranks).data / targets.size(0)
    mrr = mrr.item()
    
    return mrr


def evaluate(logits, targets, k=20):
    """ Evaluates the model using Recall@K, MRR@K scores.
    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets)
    precision = get_precision(indices,targets)
    f1 = get_f1score(recall,precision)
    mrr = get_mrr(indices, targets)
    

    return recall, precision, f1,mrr