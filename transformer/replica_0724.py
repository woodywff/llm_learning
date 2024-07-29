import math

import torch
import random

from torch import nn
from torch.nn import CrossEntropyLoss, Embedding, Dropout, Linear, LayerNorm, ReLU
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

SOS = 1
EOS = 2
PAD = 0
BATCH_SIZE = 2
LEN_SEQ_X = 15
LEN_SEQ_Y = 18
N_CLASS_X = 10
N_CLASS_Y = 11


class Attention(nn.Module):
    def __init__(self, n_head=8, n_feature=512):
        super().__init__()
        self.dim_head = n_feature // n_head
        self.n_head = n_head
        self.n_feature = n_feature

        self.fc_q = Linear(n_feature, n_feature)
        self.fc_k = Linear(n_feature, n_feature)
        self.fc_v = Linear(n_feature, n_feature)
        self.fc_o = Linear(n_feature, n_feature)

    def forward(self, q, k, v, mask):
        """

        Args:
            q: (batch size, seq_len, n_features)
            k: (batch size, seq_len, n_features)
            v: (batch size, seq_len, n_features)
            mask: src mask: (batch size, 1, 1, seq_len) or tgt mask: (batch size, 1, seq_len, seq_len)
                src mask or tgt mask for self-attention
                src mask also for cross-attention
        Returns:
            (batch size, seq_len, n_features)
        """
        # seq len for q and kv may be distinct
        batch_size, seq_len_q, _ = q.shape
        seq_len_kv = k.shape[1]
        # Linear transfer
        q = self.fc_q(q).view(batch_size, seq_len_q, self.n_head, self.dim_head).transpose(1, 2).contiguous()
        k = self.fc_k(k).view(batch_size, seq_len_kv, self.n_head, self.dim_head).transpose(1, 2).contiguous()
        v = self.fc_v(v).view(batch_size, seq_len_kv, self.n_head, self.dim_head).transpose(1, 2).contiguous()
        # Query
        # result in: (batch size, n_head, seq len q, seq len kv)
        attention_score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)
        # Mask
        attention_score = attention_score.masked_fill(mask == 0, -1e9)
        # Softmax
        attention_score = torch.softmax(attention_score, dim=-1)
        # Weighted sum
        out = torch.matmul(attention_score, v)
        # Reshape to (batch size, seq len, n feature)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.n_feature)
        # Final linear transfer
        out = self.fc_o(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_feature=512, n_hidden_feature=2048):
        super().__init__()
        self.fc_0 = Linear(n_feature, n_hidden_feature)
        self.fc_1 = Linear(n_hidden_feature, n_feature)
        self.relu = ReLU()

    def forward(self, x):
        """

        Args:
            x: (batch size, seq len, n feature)

        Returns:
            (batch size, seq len, n feature)
        """
        x = self.fc_0(x)
        x = self.relu(x)
        x = self.fc_1(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, n_feature=512, len_seq=20):
        super().__init__()
        self.pe = torch.zeros(len_seq, n_feature)
        positions = torch.arange(0, len_seq).unsqueeze(1).float()  # （seq len, 1）
        div = 1 / 10000 ** (torch.arange(0, n_feature, 2).float() / n_feature)  # (n_feature/2,)
        self.pe[:, 0::2] = torch.sin(positions * div)  # 2i
        self.pe[:, 1::2] = torch.cos(positions * div)  # 2i+1
        self.pe = self.pe.unsqueeze(0)  # (1, seq len, n_feature)

    def forward(self, x):
        """
        Args:
            x: (batch size, seq len, n_feature)

        Returns:
            (batch size, seq len, n_feature)
        """
        return x + self.pe[:, :x.shape[1]]


class Encoder(nn.Module):
    def __init__(self, n_feature=512,
                 n_hidden_feature=2048,
                 p_dropout=0.1,
                 n_head=8):
        super().__init__()
        self.attention = Attention(n_head=n_head, n_feature=n_feature)
        self.dropout = Dropout(p_dropout)
        self.norm_0 = LayerNorm(n_feature)
        self.feed_forward = FeedForward(n_feature=n_feature,
                                        n_hidden_feature=n_hidden_feature)
        self.norm_1 = LayerNorm(n_feature)

    def forward(self, x, mask_x):
        """

        Args:
            x: (batch size, seq len src, n_feature)
            mask_x: (batch size, 1, 1, seq len src)

        Returns:
            (batch size, seq len src, n_feature)
        """
        out = self.attention(x, x, x, mask_x)
        out = self.dropout(out)
        out_0 = self.norm_0(x + out)

        out = self.feed_forward(out_0)
        out = self.dropout(out)
        out = self.norm_1(out_0 + out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feature=512,
                 n_hidden_feature=2048,
                 p_dropout=0.1,
                 n_head=8):
        super().__init__()
        self.self_attention = Attention(n_head=n_head, n_feature=n_feature)
        self.dropout = Dropout(p_dropout)
        self.norm_0 = LayerNorm(n_feature)

        self.cross_attention = Attention(n_head=n_head, n_feature=n_feature)
        self.norm_1 = LayerNorm(n_feature)

        self.feed_forward = FeedForward(n_feature=n_feature,
                                        n_hidden_feature=n_hidden_feature)
        self.norm_2 = LayerNorm(n_feature)

    def forward(self, y, mask_y, encoder_output, mask_x):
        """

        Args:
            y: (batch size, seq len tgt - 1, n_feature)
            mask_y: (batch size, 1, seq len tgt -1, seq len tgt - 1)
            encoder_output: (batch size, seq len src, n_feature)
            mask_x: (batch size, 1, 1, seq len src)

        Returns:
            (batch size, seq len tgt - 1, n_feature)
        """
        out = self.self_attention(y, y, y, mask_y)
        out = self.dropout(out)
        out_0 = self.norm_0(y + out)

        out = self.cross_attention(out_0, encoder_output, encoder_output, mask_x)
        out = self.dropout(out)
        out_1 = self.norm_1(out_0 + out)

        out = self.feed_forward(out_1)
        out = self.dropout(out)
        out_2 = self.norm_2(out_1 + out)
        return out_2


class Transformer(nn.Module):
    def __init__(self, n_feature=512, n_encoder=6, n_decoder=6,
                 n_class_x=N_CLASS_X, n_class_y=N_CLASS_Y, n_hidden_feature=2048, n_head=8,
                 p_dropout=0.1):
        super().__init__()

        self.embedding_x = Embedding(n_class_x, n_feature, padding_idx=PAD)
        self.embedding_y = Embedding(n_class_y, n_feature, padding_idx=PAD)
        self.positional_encoder = PositionalEncoder(n_feature=n_feature,
                                                    len_seq=max(LEN_SEQ_X, LEN_SEQ_Y))
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
        """

        Args:
            x: (batch size, seq len src)
            mask_x: (batch size, 1, 1, seq len src)

        Returns:
            (batch size, seq len src, n_feature)
        """
        x = self.embedding_x(x)
        # (batch size, seq len src, n feature)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        for encoder in self.encoders:
            x = encoder(x, mask_x)
        return x

    def decode(self, y, mask_y, encoder_output, mask_x):
        """

        Args:
            y: tgt[:, :-1], (batch size, seq len tgt - 1)
            mask_y: (batch size, 1, seq len tgt -1, seq len tgt - 1)
            encoder_output: (batch size, seq len src, n_feature)
            mask_x: (batch size, 1, 1, seq len src)

        Returns:
            (batch size, seq len tgt - 1, n_feature)
        """
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
            (batch size, seq len tgt - 1, n_class_y)
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
