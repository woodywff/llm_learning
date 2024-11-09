import random

import torch.cuda
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from replica_0724 import Transformer
import torch

SOS = 1
EOS = 2
PAD = 0
BATCH_SIZE = 2

LEN_SEQ_X = 8
LEN_SEQ_Y = 9
N_CLASS_X = 5
N_CLASS_Y = 6


class Server:
    def __init__(self):
        super().__init__()
        self.bs = BATCH_SIZE
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src, self.tgt = self.get_data()

        self.model = Transformer(n_class_x=N_CLASS_X,
                                 n_class_y=N_CLASS_Y,
                                 device=self.device).to(self.device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.opt, factor=0.5, patience=5,
                                           verbose=True)
        self.loss = CrossEntropyLoss(ignore_index=PAD).to(self.device)

    def get_data(self):
        src = self._get_data(N_CLASS_X, LEN_SEQ_X).to(self.device)
        tgt = self._get_data(N_CLASS_Y, LEN_SEQ_Y).to(self.device)
        return src, tgt

    def _get_data(self, n_class, len_seq):
        res = []
        for _ in range(self.bs):
            # Rand data with random number of meaningful tokens
            data = torch.randint(3, n_class, (random.randint(2, len_seq - 3),))
            # Append SOS & EOS
            data = torch.cat([torch.tensor([SOS]), data, torch.tensor([EOS])])
            # Pad with PAD
            data = pad(data, (0, len_seq - len(data)), value=PAD)  # (len_seq,)
            res.append(data)
        return torch.stack(res)  # (bs, len_seq)

    def train(self, n_epoch=100):
        pbar = tqdm(range(n_epoch), desc='Training')
        for i_epoch in pbar:
            # Train
            self.model.train()
            self.opt.zero_grad()
            pred = self.model(self.src, self.tgt[:, :-1])  # (2, len_seq_y-1, n_class_y)
            loss_train = self.loss(pred.view(-1, N_CLASS_Y), self.tgt[:, 1:].reshape(-1))
            loss_train.backward()
            self.opt.step()

            # Validation
            self.model.eval()
            pred_val = self.model(self.src, self.tgt[:, :-1])
            loss_val = self.loss(pred_val.view(-1, N_CLASS_Y), self.tgt[:, 1:].reshape(-1))
            self.scheduler.step(loss_val)

            pbar.set_postfix(epoch=i_epoch,
                             loss_train=loss_train)


if __name__ == '__main__':
    server = Server()
    server.train()
