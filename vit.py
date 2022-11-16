import torch
from torch import nn
import torch.nn.functional
from torch.nn import Dropout, Softmax, Linear, LayerNorm, Conv2d
import math
import copy

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size):
        super(Embeddings, self).__init__()
        down_factor = 4
        # input image가 얼마나 많이 pooling을 거치냐가 down_factor
        # Maxpool2d가 4번 있으니 down_factor = 4
        patch_size = (2, 2)
        # patch_size는 2로 설정
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]))
        # n_pathces = (512/2**4//8) * (768/2**4//8) = 4
        self.patch_embeddings = Conv2d(in_channels=256,
                                       # 우선 in channels는 128로 설정하자
                                       out_channels=768,
                                       # out_channels = hidden size D = 768
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        # input = (B, 256, 48, 32)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # (B, 768, 24, 16)
        x = x.flatten(2)
        # (B, 768, 384)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # (B, 384, 768)
        position_embeddings = self.position_embeddings
        print(self.position_embeddings.shape)
        # position_embeddings = (B, 384, 768)
        embeddings = x + position_embeddings
        # (B, 384, 768)
        embeddings = self.dropout(embeddings)
        return embeddings


# Multi-head self attention (MSA) - layer norm not included

class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.num_attention_heads = 12
        # Number of head = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        # Attention Head size = Hidden size(D)(768) / Number of head(12) = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # All Head size = (12 * 64) = 768 = Hidden size
        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)
        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        x = x.view([x.size()[0], -1, self.num_attention_heads, self.attention_head_size])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


# MLP - layer norm not included

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Block - incorporating MSA, MLP, Layer Norm

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = MLP()
        self.attn = MSA()

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


#  ViTencoder - ViT Encoder with Blocks

class ViTencoder(nn.Module):
    def __init__(self):
        super(ViTencoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):


        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


#  ViT 마지막에 나온 latent를 CNNdecoder에 넣기 위해 변환시키기위한 Conv

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_groupnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_groupnorm),
        )
        relu = nn.LeakyReLU(inplace=True)

        gn = nn.GroupNorm(16, out_channels, eps=1e-6)

        super(Conv2dReLU, self).__init__(conv, gn, relu)


#  ViT

class ViT(nn.Module):
    def __init__(self, img_size):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = ViTencoder()
        self.img_size = img_size
        self.patch_size = (2, 2)
        self.down_factor = 4
        self.conv_more = Conv2dReLU(768, 256, kernel_size=3, padding=1, use_groupnorm=True)

    def forward(self, x):
        # (B, 256, 32, 48)
        x = self.embeddings(x)
        # (B, 384, 768)
        x = self.encoder(x)  # (B, n_patch, hidden)
        # (B, 384, 768)
        B, n_patch, hidden = x.size()
        # B=B, n_patch=384, hidden=768
        h, w = (self.img_size[0] // 2**self.down_factor // self.patch_size[0]), (self.img_size[1] // 2**self.down_factor // self.patch_size[0])
        # h=24, w=16
        x = x.permute(0, 2, 1)
        # (B, 768, 384)
        x = x.contiguous().view(B, hidden, h, w)
        # (B, 768, 16, 24)
        x = self.conv_more(x)
        # (B, 256, 16, 24)
        return x
