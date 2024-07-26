import math
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, CrossEntropyLoss, LayerNorm, ReLU
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

SOS = 0  # start of sentence
EOS = 1  # end of sentence
PAD = 2  # padding token
N_CLASS = 10
MAX_POSITION = 20
BATCH_SIZE = 2


class Transformer(nn.Module):
    def __init__(self, n_feature=512, n_encoder=6, n_decoder=6,
                 n_class=N_CLASS, n_hidden_feature=2048, n_head=8,
                 p_dropout=0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(n_class, n_feature, padding_idx=2)
        self.decoder_embedding = nn.Embedding(n_class, n_feature, padding_idx=2)

        self.positional_encoder = PositionalEncoder(n_feature=n_feature)

        self.encoders = nn.ModuleList([Encoder(n_feature=n_feature,
                                               n_hidden_feature=n_hidden_feature,
                                               p_dropout=p_dropout,
                                               n_head=n_head) for _ in range(n_encoder)])
        self.decoders = nn.ModuleList([Decoder(n_feature=n_feature,
                                               n_hidden_feature=n_hidden_feature,
                                               p_dropout=p_dropout,
                                               n_head=n_head) for _ in range(n_decoder)])

        self.final_fc = Linear(n_feature, n_class)
        self.dropout = Dropout(p_dropout)

    @staticmethod
    def get_src_mask(src):
        """
        Args:
            src: (batch size, seq_len src)
        Returns:
            src_mask: (batch size, 1, 1, seq_len src)
        """
        src_mask = (src != PAD).unsqueeze(1).unsqueeze(2).float()
        return src_mask

    @staticmethod
    def get_tgt_mask(tgt):
        """
        Args:
            tgt: (batch size, seq_len tgt - 1)
        Returns:
            tgt_mask: (batch size, 1, seq_len tgt, seq_len tgt)
        """
        tgt_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2).float()
        seq_len_tgt = tgt.shape[-1]
        diag_mask = torch.triu(torch.ones(1, seq_len_tgt, seq_len_tgt), diagonal=1) == 0
        tgt_mask = tgt_mask * diag_mask
        return tgt_mask

    def encode(self, src, src_mask):
        x = self.encoder_embedding(src)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        for encoder in self.encoders:
            x = encoder(x, src_mask)
        return x

    def decode(self, tgt, tgt_mask, encoder_output, src_mask):
        x = self.decoder_embedding(tgt)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        for decoder in self.decoders:
            x = decoder(x, tgt_mask, encoder_output, src_mask)
        return x

    def forward(self, src, tgt):

        src_mask = self.get_src_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)

        encode_output = self.encode(src, src_mask)

        x = self.decode(tgt, tgt_mask, encode_output, src_mask)

        return self.final_fc(x)


class Encoder(nn.Module):
    def __init__(self, n_feature=512, n_hidden_feature=2048, p_dropout=0.1, n_head=8):
        super().__init__()
        self.attention = Attention(n_head=n_head, n_feature=n_feature)
        self.feed_forward = FeedForward(n_feature=n_feature, n_hidden_feature=n_hidden_feature)
        self.norm_0 = LayerNorm(n_feature)
        self.norm_1 = LayerNorm(n_feature)
        self.dropout = Dropout(p_dropout)

    def forward(self, x, src_mask):
        pred_0 = self.attention(x, x, x, src_mask)
        pred_0 = self.dropout(pred_0)
        pred_0 = self.norm_0(x + pred_0)
        pred_1 = self.feed_forward(pred_0)
        pred_1 = self.dropout(pred_1)
        pred_1 = self.norm_1(pred_0 + pred_1)
        return pred_1


class Decoder(nn.Module):
    def __init__(self, n_feature=512,
                 n_hidden_feature=2048,
                 p_dropout=0.1,
                 n_head=8):
        super().__init__()
        self.self_attention = Attention(n_head=n_head, n_feature=n_feature)
        self.cross_attention = Attention(n_head=n_head, n_feature=n_feature)
        self.feed_forward = FeedForward(n_feature=n_feature, n_hidden_feature=n_hidden_feature)
        self.norm_0 = LayerNorm(n_feature)
        self.norm_1 = LayerNorm(n_feature)
        self.norm_2 = LayerNorm(n_feature)
        self.dropout = Dropout(p_dropout)

    def forward(self, x, tgt_mask, encoder_output, src_mask):
        """

        Args:
            x:  (batch size, seq_len, n_feature)
            tgt_mask: (batch size, 1, seq_len tgt, seq_len tgt)
            encoder_output: (batch size, seq_len, n_feature)
            src_mask: (batch size, 1, 1, seq_len src)

        Returns:

        """
        pred_0 = self.self_attention(x, x, x, tgt_mask)
        pred_0 = self.dropout(pred_0)
        pred_0 = self.norm_0(x + pred_0)
        pred_1 = self.cross_attention(pred_0, encoder_output, encoder_output, src_mask)
        pred_1 = self.dropout(pred_1)
        pred_1 = self.norm_1(pred_0 + pred_1)
        pred_2 = self.feed_forward(pred_1)
        pred_2 = self.dropout(pred_2)
        pred_2 = self.norm_2(pred_1 + pred_2)
        return pred_2


class PositionalEncoder(nn.Module):
    def __init__(self, n_feature=512, len_seq=MAX_POSITION):
        super().__init__()
        self.pe = torch.zeros((len_seq, n_feature))
        positions = torch.arange(0, len_seq).unsqueeze(1).float()  # (len_seq, 1)
        div = 1 / (10000 ** (torch.arange(0, n_feature, 2).float() * 2 / n_feature))  # (n_features/2)
        self.pe[:, 0::2] = torch.sin(positions * div)
        self.pe[:, 1::2] = torch.cos(positions * div)
        self.pe = self.pe.unsqueeze(0)  # (len_seq, n_features) --> (1, len_seq, n_features)

    def forward(self, x):
        """
        x: (batch size, len_seq, n_features)
        """
        return x + self.pe[:, :x.shape[1]]


class Attention(nn.Module):
    def __init__(self, n_head=8, n_feature=512):
        super().__init__()
        self.n_head = n_head
        self.n_feature = n_feature
        self.dim_head = n_feature // n_head
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

        Returns:

        """
        # (batch size, seq_len, n_features) --> (batch size, n_head, seq_len, dim_head)
        batch_size, seq_len, _ = q.shape
        q = self.fc_q(q).view(batch_size, seq_len, self.n_head, self.dim_head).transpose(1, 2).contiguous()
        seq_len_kv = k.shape[1]
        k = self.fc_k(k).view(batch_size, seq_len_kv, self.n_head, self.dim_head).transpose(1, 2).contiguous()
        v = self.fc_v(v).view(batch_size, seq_len_kv, self.n_head, self.dim_head).transpose(1, 2).contiguous()

        # (batch size, n_head, seq_len, seq_len)
        attention_score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)
        # mask
        attention_score = attention_score.masked_fill(mask == 0, -1e9)
        # (batch size, n_head, seq_len, dim_head) --> (batch size, seq_len, n_features)
        x = torch.softmax(attention_score, dim=-1)
        x = torch.matmul(x, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # Final
        return self.fc_o(x)


class FeedForward(nn.Module):
    def __init__(self, n_feature=512, n_hidden_feature=2048):
        super().__init__()
        self.fc_0 = Linear(n_feature, n_hidden_feature)
        self.fc_1 = Linear(n_hidden_feature, n_feature)
        self.relu = ReLU()

    def forward(self, x):
        x = self.fc_0(x)
        x = self.relu(x)
        x = self.fc_1(x)
        return x


def run():
    # Simulate src and tgt data
    src_list = []
    tgt_list = []
    len_src = 15
    len_tgt = 18
    n_epoch = 2

    for _ in range(BATCH_SIZE):
        # Simulate tokens of source language
        src_tokens = torch.randint(3, N_CLASS, (8,))
        # Append start and end token
        processed_src = torch.cat([torch.tensor([SOS]), src_tokens, torch.tensor([EOS])], 0)
        # Pad with pad token
        padded_src = pad(processed_src, (0, len_src - len(processed_src)), value=PAD)
        src_list.append(padded_src)

        tgt_tokens = torch.randint(3, N_CLASS, (10,))
        processed_tgt = torch.cat([torch.tensor([SOS]), tgt_tokens, torch.tensor([EOS])], 0)
        padded_tgt = pad(processed_tgt, (0, len_tgt - len(processed_tgt)), value=PAD)
        tgt_list.append(padded_tgt)

    src = torch.stack(src_list)  # (3, len_src)
    tgt = torch.stack(tgt_list)  # (3, len_tgt)

    # create model

    model = Transformer()
    loss_fn = CrossEntropyLoss(ignore_index=2)
    # Very important
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, factor=0.5,
                                           patience=5)
    # train loop
    pbar = tqdm(range(n_epoch), desc='Training')
    for i_epoch in pbar:
        model.train()
        optimizer.zero_grad()
        pred = model(src, tgt[:, :-1])
        loss = loss_fn(pred.view(-1, N_CLASS), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        model.eval()
        pred_val = model(src, tgt[:, :-1])
        loss_val = loss_fn(pred_val.view(-1, N_CLASS), tgt[:, 1:].contiguous().view(-1))
        scheduler.step(loss_val)
        pbar.set_postfix(epoch=i_epoch, loss=loss, loss_val=loss_val)

    # return

    # test loop
    y0 = torch.tensor([0])
    src_test = src[0].unsqueeze(0)
    src_mask = model.get_src_mask(src_test)
    encoder_output = model.encode(src_test, src_mask)
    pbar = tqdm(range(MAX_POSITION), desc='Test')
    for i in pbar:
        y = y0.unsqueeze(0)
        tgt_mask = model.get_tgt_mask(y)
        y = model.decode(y, tgt_mask, encoder_output, src_mask)
        y = model.final_fc(y[:, -1])  # (1, n_class)
        pred = torch.softmax(y, dim=-1)  # (3, )
        next_y0 = torch.argmax(pred, dim=-1)

        y0 = torch.cat([y0, next_y0], dim=0)
        pbar.set_postfix(step=i, output=y0)
        if next_y0 == EOS:
            break
    print(tgt[0])
    print(y0)


if __name__ == '__main__':
    run()
