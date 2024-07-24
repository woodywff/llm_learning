import torch
import random

from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from demo import Transformer

SOS = 0
EOS = 1
PAD = 2
BATCH_SIZE = 2
LEN_SEQ_X = 15
LEN_SEQ_Y = 18
N_CLASS_X = 10
# N_CLASS_Y = 11
N_CLASS_Y = 10


class Server:
    def __init__(self):
        super().__init__()
        self.bs = BATCH_SIZE
        self.src, self.tgt = self.get_data()

        self.model = Transformer(n_class=N_CLASS_X)
        self.optim = Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = ReduceLROnPlateau(self.optim, verbose=True, factor=0.5, patience=5)
        self.loss_fn = CrossEntropyLoss(ignore_index=PAD)

    def get_data(self):
        src = self._get_data(N_CLASS_X, LEN_SEQ_X)
        tgt = self._get_data(N_CLASS_Y, LEN_SEQ_Y)
        return src, tgt

    def _get_data(self, n_class, len_seq):
        data_list = []
        for _ in range(self.bs):
            # Rand data with random number of meaningful tokens
            data = torch.randint(3, n_class, (random.randint(8, 13),))
            # Append SOS & EOS
            data = torch.cat([torch.tensor([SOS]), data, torch.tensor([EOS])], dim=0)
            # Pad with PAD
            data = pad(data, (0, len_seq - len(data)), value=PAD)
            data_list.append(data)
        return torch.stack(data_list)  # (bs, len_seq)

    def train(self, n_epoch=100):

        pbar = tqdm(range(n_epoch), desc='Training')
        for i_epoch in pbar:
            self.model.train()
            self.optim.zero_grad()
            pred = self.model(self.src, self.tgt[:,:-1])    # (bs, len_seq_y-1, n_class_y)
            loss = self.loss_fn(pred.view(-1, N_CLASS_Y), self.tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            self.optim.step()

            self.model.eval()
            pred_val = self.model(self.src, self.tgt[:, :-1])  # (bs, len_seq_y-1, n_class_y)
            loss_val = self.loss_fn(pred_val.view(-1, N_CLASS_Y), self.tgt[:, 1:].contiguous().view(-1))
            self.scheduler.step(loss_val)

            pbar.set_postfix(epoch=i_epoch, loss=loss, loss_val=loss_val)

    def infer(self):
        pass


def run():
    server = Server()
    server.train()
    server.infer()


if __name__ == '__main__':
    run()
