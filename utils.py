import torch
from torch.nn.functional import log_softmax

def cross_entropy_for_onehot(pred, target):
    # Prediction should be logits instead of probs
    return torch.mean(torch.sum(-target * log_softmax(pred, dim=-1), 1))
