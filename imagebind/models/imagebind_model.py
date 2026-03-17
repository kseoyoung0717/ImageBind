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

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
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
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

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

    def _create_modality_heads(
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
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
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
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
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

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False):
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

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth", weights_only=True))

    return model
