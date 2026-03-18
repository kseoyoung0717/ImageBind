#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py ;
# https://github.com/facebookresearch/deit/blob/main/models.py
# and https://github.com/facebookresearch/vissl/blob/main/vissl/models/trunks/vision_transformer.py


from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadAttention(nn.MultiheadAttention): #기존 파이토치 클래스 상속 => 기존 기능 그대로 + forward 만 커스터마이즈 
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor): #x: 입력 토큰, attn_mask: attention 제한용 마스크
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
    #self_attention: 같은 입력끼리 비교 => query, key, value 모두 x로 설정
    #need_weights=False: attention map 반환 안함 => 메모리 절약
    #attn_mask=attn_mask: attention 제한용 마스크 전달 => 특정 토큰끼리만 attention 하도록 제한 가능
    #반환값이 (attn_output, attn_weights) 형태이므로 [0]으로 attn_output만 반환


class ViTAttention(Attention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        assert attn_mask is None
        return super().forward(x)


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = (
                x
                + self.drop_path(self.attn(self.norm_1(x), attn_mask))
                * self.layer_scale_gamma1
            )
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class SimpleTransformer(nn.Module): #여러개의 transformer block을 쌓아서 encoder 만듦 
    #input=> pre_transformer_layer(optional) => [block1, block2,...blockN] => post_taransformer_layer(optional)
    def __init__( #attention 생성 함수 
        self,
        attn_target: Callable,
        embed_dim: int,#토큰 벡터 차원(D)
        num_blocks: int, #transformer layer 개수 
        block: Callable = BlockWithMasking, #한 층(block)의 구조 
        pre_transformer_layer: Optional[Callable] = None, #transformer 앞에 붙는 레이어 
        post_transformer_layer: Optional[Callable] = None,#transformer 뒤에 붙는 레이어 
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive", #drop path 설정 
        norm_layer: Callable = _LAYER_NORM, #normalization 종류 
        mlp_ratio: int = 4, #ffn 크기 비율 
        ffn_dropout_rate: float = 0.0, 
        layer_scale_type: Optional[str] = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "jax",  # possible values jax or pytorch / weight 초기화 방식 
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer #forward에서 먼저 실행됨 
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
            #block마다 drop_path 값 다르게 / 깊을수록 더 많이 drop 
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)] #모든 block 동일 
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}") #잘못된 값 방지  

        self.blocks = nn.Sequential( #여러 블록을 순서대로 실행 
            *[
                block(
                    dim=embed_dim, #입력 차원 
                    attn_target=attn_target, #attention 설정 전달 
                    mlp_ratio=mlp_ratio, #ffn 크기 
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer, #LayerNorm 설정 
                    layer_scale_type=layer_scale_type, #LayerScale 설정
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks) #num_blocks 개수만큼 반복 
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style #나중에 초기화 방식 선택 
        self.apply(self._init_weights) #모든 서브모듈에 _init_weights 함수 적용 => weight 초기화

    def _init_weights(self, m):
        #m: 모델 안의 각 layer 
        if isinstance(m, nn.Linear): #선형 레이어일때만 실행됨 
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                #Xavier initialization: 입력/출력 분산 균형 맞춤 
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02) #truncated noraml: 값 범위 제한된 정규분포, Pytorch ViT스타일 

            if m.bias is not None:
                nn.init.constant_(m.bias, 0) #bias는 0으로 초기화
        elif isinstance(m, (nn.LayerNorm)): #layernorm일 경우 
            nn.init.constant_(m.bias, 0) #bias=0
            nn.init.constant_(m.weight, 1.0) #weight=1로 초기화 => 입력 그대로 유지하는 효과 

    def forward(
        self,
        tokens: torch.Tensor, #입력토큰
        attn_mask: torch.Tensor = None, #attention제한용 마스크
        use_checkpoint: bool = False, #메모리 절약 모드 
        checkpoint_every_n: int = 1, #몇 개의 블록마다 checkpoint 쓸지
        checkpoint_blk_ids: Optional[List[int]] = None, #특정 block만 checkpoint 
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens) #(b, l, d) => (l, b, d) 또는 layernorm
        if use_checkpoint and checkpoint_blk_ids is None: #checkpoint 사용하는데 특정 블록 지정 안했을 경우
            checkpoint_blk_ids = [ #해당 블록들에 한해서만 checkpoint 사용
                blk_id
                for blk_id in range(len(self.blocks))
                if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids) #set으로 변환해서 빠른 탐색 가능하게
        for blk_id, blk in enumerate(self.blocks): #transformer block 순회 => 각 블록마다 하나씩 실행 
            if use_checkpoint and blk_id in checkpoint_blk_ids: #checkpoint 사용하는 경우 
                #checkpoint: 메모리 절약 기술 => 원리: forward일때 중간값 저장 안하고 backward할때 다시 계산  
                tokens = checkpoint.checkpoint(
                    blk, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = blk(tokens, attn_mask=attn_mask) #그냥 일반 실행: 각 블록 안에서 attention => ffn => residual => norm 
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens) #(l, b, d) => (b, l, d) 또는 linear projection
        return tokens
