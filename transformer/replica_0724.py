import torch
import random

from torch import nn
from torch.nn import CrossEntropyLoss, Embedding, Dropout, Linear
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from demo import PositionalEncoder, Encoder, Decoder


SOS = 1
EOS = 2
PAD = 0
BATCH_SIZE = 2
LEN_SEQ_X = 15
LEN_SEQ_Y = 18
N_CLASS_X = 10
N_CLASS_Y = 11


# N_CLASS_Y = 10


#
#
class Transformer(nn.Module):
    def __init__(self, n_feature=512, n_encoder=6, n_decoder=6,
                 n_class_x=N_CLASS_X, n_class_y=N_CLASS_Y, n_hidden_feature=2048, n_head=8,
                 p_dropout=0.1):
        super().__init__()

        self.embedding_x = Embedding(n_class_x, n_feature, padding_idx=PAD)
        self.embedding_y = Embedding(n_class_y, n_feature, padding_idx=PAD)
        self.positional_encoder = PositionalEncoder(n_feature=n_feature)
        self.dropout = Dropout(p_dropout)

        self.encoders = nn.ModuleList([Encoder(n_feature=n_feature,
                                               n_hidden_feature=n_hidden_feature,
                                               p_dropout=p_dropout,
                                               n_head=n_head) for _ in range(n_encoder)])
        self.decoders = nn.ModuleList([Decoder(n_feature=n_feature,
                                               n_hidden_feature=n_hidden_feature,
                                               p_dropout=p_dropout,
                                               n_head=n_head) for _ in range(n_decoder)])
        self.fc = Linear(n_feature, n_class_y)

    @staticmethod
    def get_mask_x(x):
        """
        Get
        Args:
            x: (batch size, seq len src)

        Returns:
            (batch size, 1, 1, seq len src)
        """
        return (x != PAD).unsqueeze(1).unsqueeze(2).float()

    @staticmethod
    def get_mask_y(y):
        """

        Args:
            y: (batch size, seq len tgt - 1)

        Returns:
            (batch size, 1, seq len tgt -1, seq len tgt - 1)
        """
        mask_pad = (y != PAD).unsqueeze(1).unsqueeze(2).float()
        seq_len = y.shape[-1]
        mask_markov = torch.tril(torch.ones(seq_len, seq_len))
        # (batch size, 1, 1, seq len) * (seq len, seq len) => (batch size, 1, seq len, seq len)
        return mask_pad * mask_markov

    def encode(self, x, mask_x):
        x = self.embedding_x(x)
        # (batch size, seq len src, n feature)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        for encoder in self.encoders:
            x = encoder(x, mask_x)
        return x

    def decode(self, y, mask_y, encoder_output, mask_x):
        y = self.embedding_y(y)
        y = self.positional_encoder(y)
        y = self.dropout(y)
        for decoder in self.decoders:
            y = decoder(y, mask_y, encoder_output, mask_x)
        return y

    def forward(self, x, y):
        """

        Args:
            x: (batch size, seq len src)
            y: tgt[:, :-1], (batch size, seq len tgt - 1)
        Returns:

        """
        mask_x = self.get_mask_x(x)
        mask_y = self.get_mask_y(y)

        encoder_output = self.encode(x, mask_x)

        pred = self.decode(y, mask_y, encoder_output, mask_x)

        pred = self.fc(pred)

        return pred


class Server:
    def __init__(self):
        super().__init__()
        self.bs = BATCH_SIZE
        self.src, self.tgt = self.get_data()

        self.model = Transformer(n_class_x=N_CLASS_X, n_class_y=N_CLASS_Y)
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
            pred = self.model(self.src, self.tgt[:, :-1])  # (bs, len_seq_y-1, n_class_y)
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
