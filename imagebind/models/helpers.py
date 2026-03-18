#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import einops
import numpy as np
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__() #부모 클래스 초기화 
        self.dim = dim #정규화 할 축 저장 

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2) #L2 normalization: 벡터 길이 1로 만듦 


class LearnableLogitScaling(nn.Module):
    def __init__( 
        self,
        logit_scale_init: float = 1 / 0.07, #기본값이 14.28, CLIP에서 많이 쓰는 temperature 값=> 초기 스케일 값 의미  
        learnable: bool = True, #학습 가능한지 여부
        max_logit_scale: float = 100, #최댓값 제한 
    ) -> None:
        super().__init__() #필수, 초기화 
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable #값 저장 
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init) #로그 값을 텐서로 저장 => 힝싱 양수 유지해서 안정적인 학습 
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale) #학습 가능시 optimizer가 업데이트, gradient 계산됨 
        else:
            self.register_buffer("log_logit_scale", log_logit_scale) #학습 안하고 고정값 저장, gradient 없음, weight처럼 저장은 됨 

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x #의미: x=> x*scale 
        #exp=> scale 값으로 변환, clip=> 너무 커지는 거 제한, *x=> 벡터 스케일링 
        #similarity 값의 강도 조절 
        #exp를 통해 로그를 실제 스케일로 다시 변환

    def extra_repr(self): #pytorch에서 print할때 정보 추가 
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}," \
             f" max_logit_scale={self.max_logit_scale}" #문자열 생성 
        return st


class EinOpsRearrange(nn.Module): #nn.Module 상속/ 텐서 shape 바꾸는 레이어 
    def __init__(self, rearrange_expr: str, **kwargs) -> None: #rearrange_expr: 문자열, **kwargs:추가 옵션 
        super().__init__()
        self.rearrange_expr = rearrange_expr #변환 규칙 저장
        self.kwargs = kwargs #추가 파라미터 저장 

    def forward(self, x): #입력 텐서 x 
        assert isinstance(x, torch.Tensor) #x가 텐서인지 확인, 아닐 시 에러 발생 
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)
        #einops.rearrange: 텐서 shape 바꾸는 함수(bld => lbd)


class VerboseNNModule(nn.Module): #모델 내부를 보기 쉽게 출력하기 위한 도구 
    """
    Wrapper around nn.Module that prints registered buffers and parameter names. 
    """
    #모델 안의 parameter / buffer 정보를 출력해주는 클래스 

    @staticmethod #객체 없이도 사용 가능 
    def get_readable_tensor_repr(name: str, tensor: torch.Tensor) -> str:
        st = ( #문자열 만들기 시작 
            "("
            + name #이름 출력 
            + "): "
            + "tensor("
            + str(tuple(tensor[1].shape)) #텐서 shape 출력 
            #named_parameters()는 (name. tensor) 형태로 반환 => tensor가 index1
            + ", requires_grad="
            + str(tensor[1].requires_grad)  #gradient 여부 출력 
            + ")\n"
        )
        return st

    def extra_repr(self) -> str: #파이토치에서 print할 때 추가 정보 출력하는 함수 
        named_modules = set() #중복 제거를 위해 set 사용 
        for p in self.named_modules(): #모델안의 모든 submodule 순회 
            named_modules.update([p[0]]) #모듈 이름 저장 
        named_modules = list(named_modules) #다시 list로 변환 

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0] #이름 분리 
            if name not in named_modules: #모듈 이름 아닌 파라미터만 출력  
                string_repr += self.get_readable_tensor_repr(name, p)

        for p in self.named_buffers(): #모델안의 buffer 순회
            #buffer: 파라미터는 아니지만 저장되는 값 
            name = p[0].split(".")[0] #이름 분리 
            string_repr += self.get_readable_tensor_repr(name, p)

        return string_repr


def cast_if_src_dtype( #nn.Module 아님 , 역할: 특정 dtype 이면 다른 dtype 로 변환 (mixed precision, dtype 맞출 때 필요)
    tensor: torch.Tensor, src_dtype: torch.dtype, tgt_dtype: torch.dtype
):
    updated = False #변환 여부 기록 
    if tensor.dtype == src_dtype: #현재 텐서타입이 src_dtype이면 
        tensor = tensor.to(dtype=tgt_dtype) #실제 변환 
        updated = True #변환됨 표시 
    return tensor, updated


class QuickGELU(nn.Module): #activation function(ReLU보다 부드러운 버전)
    # From https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L166
    def forward(self, x: torch.Tensor): #입력 텐서 
        return x * torch.sigmoid(1.702 * x) #이게 quickGELU 공식인데 원래 GELU 공식보다 빠르고 성능은 비슷 


class SelectElement(nn.Module): #특정 index 토큰 뽑는 레이어 
    def __init__(self, index) -> None: #어떤 인덱스 쓸지 저장 
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3 #최소 (b, l, d) 형태 이상이어야 함 
        return x[:, self.index, ...] #(B, L, D) => L 중에서 index 번째 토큰 선택=> (B, D) 형태로 반환
        #shape 변화: (B, L, D) => (B, D)


class SelectEOSAndProject(nn.Module): #text embedding pooling 레이어 
    """
    Text Pooling used in OpenCLIP
    """

    def __init__(self, proj: nn.Module) -> None: #projection 레이어 받음 
        super().__init__()
        self.proj = proj #linear 같은 거 저장 

    def forward(self, x, seq_len): #x: (B, L, D), seq_len: (B,) => 각 시퀀스의 길이 정보(각 문장의 끝 위치)
        assert x.ndim == 3
        # x is of shape B x L x D
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), seq_len] #각 배치마다 문장의 마지막 토큰 선택 
        x = self.proj(x) #linear 변환 -> (B, D) 형태로 나옴
        return x
