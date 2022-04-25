import torch
import torch.nn.functional as F
import torch.nn as nn

class TripletLoss(nn.Module):
    "Triplet loss function"
    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_function = nn.MarginRankingLoss(margin=2)
    def forward(self, anc, pos, neg):
        dist_a_p = F.pairwise_distance(anc, pos, 2)
        dist_a_n = F.pairwise_distance(anc, neg, 2)
        target = torch.FloatTensor(dist_a_p.size()).fill_(1)
        target = target.cuda()
        return self.loss_function(dist_a_n, dist_a_p, target)