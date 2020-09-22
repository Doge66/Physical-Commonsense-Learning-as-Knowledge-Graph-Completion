import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckERNoDropoutBN(torch.nn.Module):
    def __init__(self, d, d1, d2, cuda):
        super(TuckERNoDropoutBN, self).__init__()

        self.d1 = d1
        self.d2 = d2

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda" if cuda==True else "cpu", requires_grad=True))
        

        self.loss = torch.nn.BCELoss()
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, e2_idx):
        e1 = self.E(e1_idx)
        x = e1.view(-1, 1, e1.size(1))

        e2 = self.E(e2_idx)
        W_mat = torch.mm(e2, self.W.view(e2.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), self.d2)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, self.d2)
        x = torch.mm(x, self.R.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return (pred, self.W, self.E.weight, self.R.weight)

    def forward_lp(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = e1.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return (pred, self.W, self.E.weight, self.R.weight)
