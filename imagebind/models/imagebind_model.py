#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.




import os
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from imagebind.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from imagebind.models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from imagebind.models.transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class ImageBindModel(nn.Module):
    def __init__( #input -> preprocessing -> transformer -> projection -> normalize 
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768, #vision token dimension   
        vision_embed_dim=1024,
        vision_num_blocks=24, #num_blocks=> transformer layer 수 
        vision_num_heads=16, #num_heads=> attention head 수 
        audio_embed_dim=768, #audio token dimension 
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768, #text token dimension 
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            #모달리티별 입력 데이터를 token으로 변환하는 부분 
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            #모달리티마다 다른 transformer encoder를 하나씩 만들어서 저장 
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            #각 모달리티의 head 레이어들을 생성, 즉 transformer가 만든 feature를 최종 embedding dimension으로 projection하는 부분  
            #필요 이유:각 modality transformer의 출력 dimension이 다른데 모델의 최종목표는 모든 modality를 같은 embedding dimension으로 통일하는 것이므로 
            out_embed_dim, #최종 embedding dimension
            vision_embed_dim,#vision transformer 출력 dimension 
            text_embed_dim, #text transformer 출력 dimension
            audio_embed_dim, #audio transformer 출력 dimension
            depth_embed_dim, #depth transformer 출력 dimension
            thermal_embed_dim, #thermal transformer 출력 dimension
            imu_embed_dim, #imu transformer 출력 dimension
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )#embedding dimension을 normalize하는 부분 

    def _create_modality_preprocessors(
        #각 모달리티 입력을 transformer가 처리할 수 있는 token 형태로 변환하는 preprocessor 모듈들을 생성하기 위함
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric( #vison 입력을 patch token으로 변환하는 stem 모듈을 생성 
            proj_stem=[ #실행순서: PamIm2Video => Conv3d => patch embedding 
                PadIm2Video(pad_type="repeat", ntimes=2), #이미지를 비디오 형태로 변환: 입력형태를 [C, H, W]=> [C, T, H, W]형태로 만들어야함 
                #ntimes=2: 같은 이미지를 2frame으로 반복 
                nn.Conv3d( #patch embedding 역할을 함 
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size, #patch단위로 입력을 쪼갬 
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor( #vision 모달리티 전체 preprocessing 모듈을 생성(stem + cls token + positional embedding)
            img_size=[3, video_frames, 224, 224], #입력 shape: [C, T, H, W]
            num_cls_tokens=1, #transformer에서 사용하는 cls token 개수 
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), #position embedding 생성 함수
            #vision token은 time + height + width ->위치 정보를 가지므로 spatiotemporal positional embedding을 사용함 
            #learnable=True: position embedding이 학습가능하도록 설정 
            rgbt_stem=rgbt_stem,
            depth_stem=None, #이 preprocessor에서는 depth 입력을 사용하지 않음 => none 
        )

        text_preprocessor = TextPreprocessor( #텍스트 입력을 transformer 에 넣을 수 있는 token embedding으로 변환하는 전처리 모듈 생성 
            context_length=77, #텍스트 최대 토큰 길이 => 한 문장은 최대 77token까지만 사용됨

            #토큰화: 데이터를 잘게 나눈 뒤, 각 조각을 숫자 벡터로 바꾸는 과정

            vocab_size=49408, #사용하는 vocabulary 크기 => tokenizer 가 사용하는 단어(토큰)개수 
            embed_dim=text_embed_dim, #각 토큰을 변환할 embedding vector dimension 
            causal_masking=True, #transformer attention 에서 causal mask를 사용한다는 의미, 즉 attention이 현재 토큰->이전 토큰만 볼 수 있음을 의미 
        )

        audio_stem = PatchEmbedGeneric( #오디오 입력을 patch token으로 변환하는 stem 모듈을 생성 
                                       #오디오는 보통 mel spectrogram 형태로 입력됨 => [1, mel_bins, audio_target_len] 형태
            proj_stem=[
                nn.Conv2d( #audio spectrogram을 patch 단위로 나누고 embedding변환 
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim, #각 patch가 audio_embed_dim 차원의 벡터가 됨 
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim), #patch embedding이후에 LayerNorm을 적용함 
        )
        audio_preprocessor = AudioPreprocessor( #audio 전체 preprocessing 모듈을 생성함 
            img_size=[1, audio_num_mel_bins, audio_target_len], #오디오 입력 데이터 형태 => [channels, mel_bins, time_steps]
            num_cls_tokens=1, #transformer에서 사용하는 cls token 개수 => cls+audio patch token형태로 들어감 
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True), #patch token에 position embedding을 추가하는 함수 
            #audio spectrogram은 frequency + time-> 위치정보를 가지므로 positional embedding 필요 
            #learnable=True: position embedding이 학습가능하도록 설정 
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks( #각 모달리티별 (전용 인코더)transformer trunk를 생성하는 부분
        #trunk: 입력을 받아 feature로 바꾸는 '메인 인코더 네트워크' 
        #진행순서: raw input => trunk => embedding(벡터)
        self,
        vision_embed_dim=1024,#featue 차원 
        vision_num_blocks=24, #transformer layer 수
        vision_num_heads=16, #multi-head attention 개수 
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk( #주어진 설정값으로 실제 transformer 모델 생성
            #설정값을 받아서 transformer 인코더 하나를 생성해서 반환하는 함수 
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim, #각 토큰의 벡터 크기 
                num_blocks=num_blocks, #transformer layer 수
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path, #regularization(layer 일부를 랜덤으로 스킵)=> 과적합 방지, 깊은 모델 안정화
                attn_target=partial( #partial 함수 쓰는 이유: 미리 설정된 attention 생성기를 만듦 
                    MultiheadAttention, # Q, K, V=> attention => weighted sum 
                    embed_dim=embed_dim,
                    num_heads=num_heads, #attention head 수 => multi-head attention 구현(하나의 attention 여러 개로 쪼개서 병렬 처리)
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) #토큰 벡터 정규화 => 학습 안정화 
                    if pre_transformer_ln 
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"), #차원 순서 바꾸기(보통 transformer 내부는 (sequence length, batch, dim) 형태를 기대하는데 현재 입력은 batch, length, dim =>처리하기 편하게 변경 )
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"), #다시 원래 순서로 복구 
            )
            #trunk 내부 흐름: 입력 토큰 (b, l, d)=> LayerNorm(optional) => transformer blocks(attention + FFN)=> (l, b, d)->(b, l, d) => 출력 
            #attention: 다른 토큰 참고해서 정보 섞기(각 토큰이 다른 토큰들을 얼마나 참고할지 결정)
                #q(질문), k(키), v(값) => attention score 계산: SOFTMAX(QK^T)V => 가중합 => 토큰 벡터 업데이트
            #FFN: 각 토큰을 개별적으로 더 복잡하게 변환: 토큰끼리 안섞이고 독립처리 

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads( #trunk에서 나온 토큰들을 최종 embedding으로 바꾸는 마지막 변환 
        #전체 흐름이 입력 => 토큰화 => trunk(transformer) => head(최종 embedding) => postprocessor(normalize) 이므로
        #head는 trunk에서 나온 토큰들을 최종 embedding dimension으로 projection하는 역할을 함
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6), #LayerNorm: 토큰 벡터 정규화, 값 분포 안정화하는 역할 
            #layernorm사용하는 이유: embedding quality 안정화, modality간 비교 잘되도록 하기 위해서 
            SelectElement(index=0), #결과가 patch token + cls token 형태로 나오는데, cls token(첫번째 토큰=>index=0)만 선택해서 최종 embedding으로 사용
            #cls token: transformer에서 전체 입력 시퀀스의 정보를 압축해서 담는 특별한 토큰 => 최종적으로 이 토큰이 전체 시퀀스의 대표 embedding이 됨
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False), #nn.liner: 차원 맞추기 + embedding 공간 정렬 
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim): 
        #postprocessor: embedding 정규화, 필요시 스케일 조정해서 비교하기 좋게 만드는 단계 
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1) 
        #normalize(dim=-1): L2정규화 => 벡터의 크기를 1로 만들어서 방향성 정보만 남김 => cosine similarity 계산할 때 유리
        #이렇게 정규화하는 것의 핵심 효과: 방향만 비교, 크기는 무시 
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True) #learnablelogitscaling: similarity 값의 세기 조절
            #similarity=x*y 이걸 scaled=similarity * scale 로 변경 (그냥 값으로 했을 때 너무 작으면 구분이 어렵고, 너무 크면 gradient가 불안정하기 때문에)
            #vision에만 없는 이유: vision을 기준(anchor)로 두고 나머지 모달리티를 거기에 맞추기 위해 scaling 
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs): #모달리티별 입력을 받아서 최종 embedding으로 변환하는 부분(전체 흐름)
        outputs = {}
        for modality_key, modality_value in inputs.items(): #루프: 모달리티별 처리 의미 
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips(오디오, 비디오는 클립 여러개로 들어옴)
            if reduce_list: #아래와 같이 reshape 하는 이유: clip들을 batch처럼 한꺼번에 처리하기 위해서 
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key]( #preprocessor=> 여기서 텍스트 토큰화, 이미지 patch분할, 오디오 spectrogram 변환 
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"] #trunk입력 
                head_inputs = modality_value["head"] 
                modality_value = self.modality_trunks[modality_key](**trunk_inputs) #attention + FFN 반복 => 토큰 간 정보 섞임 
                modality_value = self.modality_heads[modality_key]( #여기서 cls token 선택 + linear projection 
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key]( #postprocessor => L2 normalization + vision제외 scaling 
                    modality_value
                )

                if reduce_list: #처리한 clip 여러개를 평균내서 하나의 embedding으로 만듦 
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value #output에 저장 

        return outputs


def imagebind_huge(pretrained=False): #imagebind 큰 버전 모델 만들고 필요 시 학습된 가중치 불러오는 함수 
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained: #true면 학습된 weight 불러옴 
        if not os.path.exists(".checkpoints/imagebind_huge.pth"): #weight 파일 없으면 다운로드, 로컬에 없을 시  
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth", #로컬에 없을 시 이 url에서 다운로드 
                ".checkpoints/imagebind_huge.pth", #저장위치 
                progress=True,
            )

        model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth", weights_only=True))
        #.pth 파일 = 모델 파라미터, 그걸 현재 모델 구조에 넣음 => random model에서 pretrained 모델이 됨 

    return model
