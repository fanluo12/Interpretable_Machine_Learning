########################### import ###############################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# class Encoder
class Encoder(nn.Module):
    # "Encoder is the stack of N Encoder-layer"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        # "deal line by line"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# layers connection
class LayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(LayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x


class EncoderLayer(nn.Module):
    # "connect atttention layer and feedforward layer"
    def __init__(self, size, en_attention, en_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.en_attention = en_attention
        self.en_feedforward = en_feedforward
        self.dropout = dropout
        self.size = size

    def forward(self, x):
        x = self.en_attention(x)
        x = self.en_feedforward(x)

        return x


# class encoderlayer attention

class EncoderLayer_attention(nn.Module):
    # "EncoderLayer consists of self-atten and feed forward"
    def __init__(self, size, self_attn, dropout):
        super(EncoderLayer_attention, self).__init__()
        self.self_attn = self_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
        self.dropout = dropout

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return x


# class encoderlayer feedfor

class EncoderLayer_feedfor(nn.Module):
    # "EncoderLayer consists of self-atten and feed forward"
    def __init__(self, size, feed_forward, dropout):
        super(EncoderLayer_feedfor, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
        self.dropout = dropout

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.feed_forward(x))
        return x


# class attention_last
class Attention_last(nn.Module):
    # "the last attention layer"
    def __init__(self, size, self_attn, dropout):
        super(Attention_last, self).__init__()
        self.self_attn = self_attn
        self.size = size
        self.dropout = dropout

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x


# class LayerNorm

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# sub-layers connection

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# attention
# multiheaded attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=None,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        rs = self.linears[-1](x)
        return rs


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# positional encodding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



class wCNN(nn.Module):
    def __init__(self):
        super(wCNN, self).__init__()
        self.num_filters = 20
        self.dim = 512
        self.ngram_sizes = [2, 3]
        self.num_classes = 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, self.num_filters, self.dim * window_size, stride=self.dim, bias=True)
             for window_size in self.ngram_sizes]
        )
        self.fc = nn.Linear(self.num_filters * len(self.ngram_sizes), self.num_classes)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[1] * x.shape[2])
        out = {}
        features = [conv(x) for conv in self.convs]
        features = [F.relu(v) for v in features]
        pooled = [F.max_pool1d(feat, feat.size()[2]).view(-1, feat.shape[1]) for feat in features]
        pooled = torch.cat(pooled, 1)
        logit = self.fc(pooled)
        logit = F.softmax(logit, 1)
        out['logits'] = logit
        return logit

class Softmax(nn.Module):
    def __init__(self,D_w):
        super(Softmax,self).__init__()
        self.Dim = 2
        self.D_w = D_w
        self.matrix_w = nn.Parameter(torch.ones(1,self.D_w))
    def forward(self,x):
        x = torch.matmul(self.matrix_w,x)
        x = F.softmax(x,self.Dim)
        return x

class wCNN_pos(nn.Module):
    def __init__(self):
        super(wCNN_pos,self).__init__()
        self.num_filters = 20
        self.dim = 1
        self.ngram_sizes = [1]
        self.num_classes = 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, self.num_filters, self.dim * window_size, stride=self.dim, bias=True)
             for window_size in self.ngram_sizes]
        )
        self.fc = nn.Linear(self.num_filters * len(self.ngram_sizes), self.num_classes)

    def forward(self,x):
        out = {}
        features = [conv(x) for conv in self.convs]
        features = [F.relu(v) for v in features]
        pooled = [F.max_pool1d(feat, feat.size()[2]).view(-1, feat.shape[1]) for feat in features]

        pooled = torch.cat(pooled, 1)
        logit = self.fc(pooled)
        logit = F.softmax(logit,1)
        out['logits'] = logit
        return out['logits']

class wCNN_cat(nn.Module):
    def __init__(self):
        super(wCNN_cat, self).__init__()
        self.num_filters = 1
        self.dim = 2
        self.ngram_sizes = [1]
        self.num_classes = 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, self.num_filters, self.dim * window_size, stride=self.dim, bias=True)
             for window_size in self.ngram_sizes]
        )
        self.fc = nn.Linear(self.num_filters * len(self.ngram_sizes), self.num_classes)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[1] * x.shape[2])
        out = {}
        features = [conv(x) for conv in self.convs]
        features = [F.relu(v) for v in features]
        pooled = [F.max_pool1d(feat, feat.size()[2]).view(-1, feat.shape[1]) for feat in features]
        pooled = torch.cat(pooled, 1)
        logit = self.fc(pooled)
        logit = F.softmax(logit, 1)
        out['logits'] = logit
        return logit

class Dense(nn.Module):
    def __init__(self):
        super(Dense,self).__init__()
        self.w_eye = nn.Parameter(torch.eye(2))
    def forward(self,x):
        x_n = []
        x_ = x.view(x.size(0),-1)
        x_ = F.softmax(x_,1)
        x_n = x_.data.numpy()
        x = torch.matmul(self.w_eye,x)
        x = torch.sum(x,1,True)
        x = F.softmax(x,2)
        return x

def make_model_pos(N=6,d_model=36,d_ff=36*4,h=1,dropout=0.01):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h,d_model)
    attn_last = MultiHeadedAttention(1,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    model = nn.Sequential(
            Encoder(
                    EncoderLayer(
                            d_model,
                            EncoderLayer_attention(d_model,c(attn),dropout),
                            EncoderLayer_feedfor(d_model,c(ff),dropout),
                            dropout),
                            N),
            Attention_last(d_model,c(attn_last),dropout),
            wCNN_pos(),
            )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)
    return model


def make_model_embed(D_w, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn_last = MultiHeadedAttention(1, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = nn.Sequential(c(position),
                          Encoder(
                              EncoderLayer(
                                  d_model,
                                  EncoderLayer_attention(d_model, c(attn), dropout),
                                  EncoderLayer_feedfor(d_model, c(ff), dropout),
                                  dropout),
                              N),

                          Attention_last(d_model, c(attn_last), dropout),
                          wCNN()
                          )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def make_connect(D_w=2, N=6, d_model=2, d_ff=2*4, h=1, dropout=0.1):
    c = copy.deepcopy
    attn_last = MultiHeadedAttention(1, d_model)
    model = nn.Sequential(
        Attention_last(d_model, c(attn_last), dropout),

        Dense()
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model



