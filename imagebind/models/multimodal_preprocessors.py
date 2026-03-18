#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import html
import io
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from timm.layers import trunc_normal_

from imagebind.models.helpers import VerboseNNModule, cast_if_src_dtype


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    # nn.functional.interpolate doesn't work with bfloat16 so we cast to float32
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch.float32)
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
            0, 3, 1, 2
        ),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed


def interpolate_pos_encoding(
    npatch_per_img,
    pos_embed,
    patches_layout,
    input_shape=None,
    first_patch_idx=1,
):
    assert first_patch_idx == 0 or first_patch_idx == 1, "there is 1 CLS token or none"
    N = pos_embed.shape[1] - first_patch_idx  # since it's 1 if cls_token exists
    if npatch_per_img == N:
        return pos_embed

    assert (
        patches_layout[-1] == patches_layout[-2]
    ), "Interpolation of pos embed not supported for non-square layouts"

    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]

    if input_shape is None or patches_layout[0] == 1:
        # simple 2D pos embedding, no temporal component
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
    elif patches_layout[0] > 1:
        # pos embed has a temporal component
        assert len(input_shape) == 4, "temporal interpolation not supported"
        # we only support 2D interpolation in this case
        num_frames = patches_layout[0]
        num_spatial_tokens = patches_layout[1] * patches_layout[2]
        pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
        # interpolate embedding for zeroth frame
        pos_embed = interpolate_pos_encoding_2d(
            npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0)
        )
    else:
        raise ValueError("This type of interpolation isn't implemented")

    return torch.cat((class_emb, pos_embed), dim=1)


def _get_pos_embedding(
    npatch_per_img,
    pos_embed,
    patches_layout,
    input_shape,
    first_patch_idx=1,
):
    pos_embed = interpolate_pos_encoding(
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=input_shape,
        first_patch_idx=first_patch_idx,
    )
    return pos_embed


class PatchEmbedGeneric(nn.Module): #이미지를 잘게 patch로 쪼개고 transformer가 받을 수 있는 토큰 시퀀스로 변환 
    """
    PatchEmbed from Hydra
    """
    #Hydra 구조에서 가져온 patch embedding 방식 

    def __init__(self, proj_stem, norm_layer: Optional[nn.Module] = None):
        #proj_stem: 이미지를 feature 변환 레이어들로 변환 
        #norm_layer: optional normalization 
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem) #여러 레이어면 하나로 묶음(conv => conv => conv..이런식)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0] #레이어 하나면 그대로 사용 
        #이렇게 하는 이유: pretrained 모델 호환성 때문 
        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        #image size: [c, h, w] 또는 [c, t, h, w] 형태의 튜플
        with torch.no_grad(): #계산만 하고 gradient 안씀 
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            ) #테스트용 입력: (1, ㅊ, h, w)
            dummy_out = self.proj(dummy_img) #conv 등 적용됨 
        embed_dim = dummy_out.shape[1] #채널 수= 임베딩 차원 
        patches_layout = tuple(dummy_out.shape[2:]) #spatial 구조 
        num_patches = np.prod(patches_layout) #전체 패치 수 
        return patches_layout, num_patches, embed_dim

    def forward(self, x): #입력: x => (b, c, h, w) 또는 (b, c, t, h, w)
        x = self.proj(x) #보통 conv layer, 또는 linear patch embedding 
        # B C (T) H W -> B (T)HW C
        x = x.flatten(2).transpose(1, 2) #의미: (b, c, h, w) => (b, c, h*w) => (b, h*w, c) => transformer가 받을 수 있는 토큰 시퀀스 형태로 변환
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class SpatioTemporalPosEmbeddingHelper(VerboseNNModule): 
    def __init__(
        self,
        patches_layout: List,
        num_patches: int, #전체 patch 개수 
        num_cls_tokens: int, #cls tocken 개수 
        embed_dim: int, #embedding 차원 
        learnable: bool, #positional embedding을 학습할지 여부 
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens #cls 개수 저장 
        self.patches_layout = patches_layout #patch 구조 저장 
        self.num_patches = num_patches #patch 개수 
        self.num_tokens = num_cls_tokens + num_patches #전체 토큰 수(cls + patch들)
        self.learnable = learnable #학습여부 저장 
        if self.learnable:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            #shape가 (1, num_tockens, embed_dim) => 학습 가능한 파라미터 
            trunc_normal_(self.pos_embed, std=0.02) #초기화(작은 랜덤값)
        else:
            self.register_buffer( #register_buffer: weight는 아니지만 모델에 저장됨, gradient 없음 
                "pos_embed", get_sinusoid_encoding_table(self.num_tokens, embed_dim)
                #sinusoidal positioning encoding => 학습 안함, 수학적으로 생성 
            )

    def get_pos_embedding(self, vision_input, all_vision_tokens): #입력: vision_input=> 원본 이미지, all_vision_tockens=> 토큰들(cls 포함)
        input_shape = vision_input.shape
        pos_embed = _get_pos_embedding( #실제 계산 담당 함수 
            all_vision_tokens.size(1) - self.num_cls_tokens, #patch 개수만 계산 
            pos_embed=self.pos_embed, #앞에서 만든 임베딩 
            patches_layout=self.patches_layout, #실제 입력 크기
            input_shape=input_shape, 
            first_patch_idx=self.num_cls_tokens, #cls 이후부터 patch 시작  
        )
        return pos_embed


class RGBDTPreprocessor(VerboseNNModule):
    def __init__(
        self,
        rgbt_stem: PatchEmbedGeneric,
        depth_stem: Optional[PatchEmbedGeneric],
        img_size: Tuple = (3, 224, 224),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        use_type_embed: bool = False,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        stem = rgbt_stem if rgbt_stem is not None else depth_stem
        (
            self.patches_layout,
            self.num_patches,
            self.embed_dim,
        ) = stem.get_patch_layout(img_size)
        self.rgbt_stem = rgbt_stem
        self.depth_stem = depth_stem
        self.use_pos_embed = pos_embed_fn is not None
        self.use_type_embed = use_type_embed
        self.num_cls_tokens = num_cls_tokens

        if self.use_pos_embed:
            self.pos_embedding_helper = pos_embed_fn(
                patches_layout=self.patches_layout,
                num_cls_tokens=num_cls_tokens,
                num_patches=self.num_patches,
                embed_dim=self.embed_dim,
            )
        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )
        if self.use_type_embed:
            self.type_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.use_pos_embed:
                nn.init.normal_(self.pos_embedding_helper.pos_embed)
                self.pos_embedding_helper.pos_embed *= scale

            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

        if self.use_type_embed:
            nn.init.normal_(self.type_embed)

    def tokenize_input_and_cls_pos(self, input, stem, mask):
        # tokens is of shape B x L x D
        tokens = stem(input)
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            pos_embed = self.pos_embedding_helper.get_pos_embedding(input, tokens)
            tokens = tokens + pos_embed
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(B, -1, -1)
        return tokens

    def forward(self, vision=None, depth=None, patch_mask=None):
        if patch_mask is not None:
            raise NotImplementedError()

        if vision is not None:
            vision_tokens = self.tokenize_input_and_cls_pos(
                vision, self.rgbt_stem, patch_mask
            )

        if depth is not None:
            depth_tokens = self.tokenize_input_and_cls_pos(
                depth, self.depth_stem, patch_mask
            )

        # aggregate tokens
        if vision is not None and depth is not None:
            final_tokens = vision_tokens + depth_tokens
        else:
            final_tokens = vision_tokens if vision is not None else depth_tokens
        return_dict = {
            "trunk": {
                "tokens": final_tokens,
            },
            "head": {},
        }
        return return_dict


class AudioPreprocessor(RGBDTPreprocessor):
    def __init__(self, audio_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=audio_stem, depth_stem=None, **kwargs)

    def forward(self, audio=None):
        return super().forward(vision=audio)


class ThermalPreprocessor(RGBDTPreprocessor):
    def __init__(self, thermal_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=thermal_stem, depth_stem=None, **kwargs)

    def forward(self, thermal=None):
        return super().forward(vision=thermal)


def build_causal_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


class TextPreprocessor(VerboseNNModule):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.context_length + num_cls_tokens, embed_dim)
        )
        self.causal_masking = causal_masking
        if self.causal_masking:
            mask = build_causal_attention_mask(self.context_length)
            # register the mask as a buffer so it can be moved to the right device
            self.register_buffer("mask", mask)

        self.supply_seq_len_to_head = supply_seq_len_to_head
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        if num_cls_tokens > 0:
            assert self.causal_masking is False, "Masking + CLS token isn't implemented"
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, embed_dim)
            )

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style="openclip"):
        # OpenCLIP style initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def forward(self, text):
        # text tokens are of shape B x L x D
        text_tokens = self.token_embedding(text)
        # concat CLS tokens if any
        if self.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            text_tokens = torch.cat((class_tokens, text_tokens), dim=1)
        text_tokens = text_tokens + self.pos_embed
        return_dict = {
            "trunk": {
                "tokens": text_tokens,
            },
            "head": {},
        }
        # Compute sequence length after adding CLS tokens
        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(dim=-1)
            return_dict["head"] = {
                "seq_len": text_lengths,
            }
        if self.causal_masking:
            return_dict["trunk"].update({"attn_mask": self.mask})
        return return_dict


class Im2Video(nn.Module):
    """Convert an image into a trivial video."""

    def __init__(self, time_dim=2):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x):
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            return x.unsqueeze(self.time_dim)
        elif x.ndim == 5:
            return x
        else:
            raise ValueError(f"Dimension incorrect {x.shape}")


class PadIm2Video(Im2Video): #이미지를 시간축(t)를 가진 데이터로 바꾼 뒤 부족한 프레임을 복제하거나 0으로 채워서 영상처럼 만드는 클래스
    def __init__(self, ntimes, pad_type, time_dim=2):
        #ntimes: 최종 time 길이(프레임 수), pad_type: zero또는 repeat, time_dim: 시간 축 위치
        super().__init__(time_dim=time_dim) #Im2Video=> 보통 (b, c, h, w)->(b, c, t=1, h, w)로 바꿔줌 
        assert ntimes > 0 #프레임 수는 양수여야 함 
        assert pad_type in ["zero", "repeat"] #pad 방식 제한 
        self.ntimes = ntimes
        self.pad_type = pad_type #값 저장 

    def forward(self, x): #입력 x: 이미지 또는 영상 / 영상일때만 t 있음 
        x = super().forward(x) #결과: 시간축 t=1 생성 
        if x.shape[self.time_dim] == 1: #시간 길이 1이면 padding 수행 
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes #시간축만 ntimes로 설정 
                x = x.repeat(new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape) #padding인자 초기화 
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim] #시간축 뒤쪽에 Padding 추가 
                x = nn.functional.pad(x, padarg) #zero padding 적용 
        return x
    #imagebind는 image, video, audio 모두 처리하기에 입력형태 통일 필요 => 이미지를 가짜 비디오로 변환(멀티모달 입력형식 통일용)


# Modified from github.com/openai/CLIP
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str, context_length=77):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with g_pathmgr.open(bpe_path, "rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merges: List[str] = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges: List[Tuple[str, ...]] = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.context_length = context_length

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(self, texts, context_length=None):
        if not context_length:
            context_length = self.context_length

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:context_length - 1] + [eot_token]
            result[i, : len(tokens)] = torch.tensor(tokens)

        if len(result) == 1:
            return result[0]
        return result


class IMUPreprocessor(VerboseNNModule):
    def __init__(
        self,
        kernel_size: int,
        imu_stem: PatchEmbedGeneric,
        embed_dim: int,
        img_size: Tuple = (6, 2000),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.imu_stem = imu_stem
        self.embed_dim = embed_dim
        self.use_pos_embed = pos_embed_fn is not None
        self.num_cls_tokens = num_cls_tokens
        self.kernel_size = kernel_size
        self.pos_embed = nn.Parameter(
            torch.empty(1, (img_size[1] // kernel_size) + num_cls_tokens, embed_dim)
        )

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5

            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def tokenize_input_and_cls_pos(self, input, stem):
        # tokens is of shape B x L x D
        tokens = stem.norm_layer(stem.proj(input))
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        return tokens

    def forward(self, imu):
        # Patchify
        imu = imu.unfold(
            -1,
            self.kernel_size,
            self.kernel_size,
        ).permute(0, 2, 1, 3)
        imu = imu.reshape(imu.size(0), imu.size(1), -1)

        imu_tokens = self.tokenize_input_and_cls_pos(
            imu,
            self.imu_stem,
        )

        return_dict = {
            "trunk": {
                "tokens": imu_tokens,
            },
            "head": {},
        }
        return return_dict
