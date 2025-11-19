from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import logging

from .configuration_qwen3_omni import (
    Qwen3OmniConfig,
    Qwen3OmniThinkerConfig,
    Qwen3OmniTalkerConfig,
)

logger = logging.get_logger(__name__)


class Qwen3OmniPreTrainedModel(PreTrainedModel):
    """
    Qwen3 Omni 所有子模型的基类。
    类似 Qwen2_5OmniPreTrainedModel。
    """

    config_class = Qwen3OmniConfig
    base_model_prefix = "qwen3_omni"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # TODO: 根据 Qwen3 的初始化策略修改
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class Qwen3OmniAudioEncoder(Qwen3OmniPreTrainedModel):
    config_class = Qwen3OmniThinkerConfig  # 内部用 thinker_config.audio_config

    def __init__(self, config: Qwen3OmniThinkerConfig):
        super().__init__(config)
        audio_config = config.audio_config
        # TODO: 实现真正的 audio encoder 堆叠
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=audio_config.hidden_size,
                    nhead=audio_config.num_attention_heads,
                    batch_first=True,
                )
                for _ in range(audio_config.num_hidden_layers)
            ]
        )

    def forward(self, input_features: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # TODO: 实现真实的 mask 处理 & positional encoding
        x = input_features
        for layer in self.layers:
            x = layer(x)
        return x  # [batch, seq, hidden_size]


class Qwen3OmniVisionEncoder(Qwen3OmniPreTrainedModel):
    config_class = Qwen3OmniThinkerConfig

    def __init__(self, config: Qwen3OmniThinkerConfig):
        super().__init__(config)
        vision_config = config.vision_config
        # TODO: 实现真正的 vision patch embedding + transformer encoder
        self.proj = nn.Linear(vision_config.hidden_size, vision_config.out_hidden_size)

    def forward(self, pixel_values: Tensor, grid_thw: Optional[Tensor] = None) -> Tensor:
        # TODO: 把 pixel_values -> patch embedding -> transformer -> 输出统一 hidden
        x = self.proj(pixel_values)
        return x


class Qwen3OmniThinkerTextModel(Qwen3OmniPreTrainedModel):
    config_class = Qwen3OmniThinkerConfig

    def __init__(self, config: Qwen3OmniThinkerConfig):
        super().__init__(config)
        text_config = config.text_config
        # TODO: 这里最好直接用 Qwen3Config 对应的模型（比如 Qwen3Model），目前先写非常简化的 skeleton
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=text_config.hidden_size,
                    nhead=text_config.num_attention_heads,
                    batch_first=True,
                )
                for _ in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(text_config.hidden_size)

    def forward(self, input_ids: Optional[Tensor] = None, inputs_embeds: Optional[Tensor] = None, **kwargs) -> Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Qwen3OmniThinkerForConditionalGeneration(Qwen3OmniPreTrainedModel, GenerationMixin):
    config_class = Qwen3OmniThinkerConfig
    base_model_prefix = "thinker"

    def __init__(self, config: Qwen3OmniThinkerConfig):
        super().__init__(config)
        self.config = config

        self.audio_tower = Qwen3OmniAudioEncoder(config)
        self.visual = Qwen3OmniVisionEncoder(config)
        self.model = Qwen3OmniThinkerTextModel(config)

        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.pad_token_id = config.text_config.pad_token_id if config.text_config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.patch_size  # TODO: 根据 Qwen3 视觉结构调整
        self.rope_deltas = None

    def get_position_id_func(self):
        """
        TODO: 参考 Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func，
        实现 Qwen3 Omni 的 position id / rope index 计算接口。
        """
        def dummy_position_id_func(input_ids, image_grid_thw=None, video_grid_thw=None, feature_lengths=None, **kwargs):
            # 先给一个最简 placeholder，后续根据 Qwen3 真实策略实现
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            return {"position_ids": position_ids}

        return dummy_position_id_func

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        audio_features: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # TODO: 根据 Qwen2.5 的逻辑，把 audio / vision 特征和文本序列融合起来
        hidden_states = self.model(input_ids=input_ids)

        # 简化：直接算 logits，不做多模态融合（placeholder）
        logits = self.lm_head(hidden_states)
        return logits


class Qwen3OmniTalkerForConditionalGeneration(Qwen3OmniPreTrainedModel, GenerationMixin):
    config_class = Qwen3OmniTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen3OmniTalkerConfig):
        super().__init__(config)
        self.config = config
        # TODO: 实现真正的 TTS 模型结构（Transformer/AR decoder + codec head）
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    batch_first=True,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        logits = self.codec_head(x)
        return logits


class Qwen3OmniForConditionalGeneration(Qwen3OmniPreTrainedModel, GenerationMixin):
    """
    顶层门面：包含 Thinker（文本/多模态）和可选 Talker（TTS）。
    """

    config_class = Qwen3OmniConfig

    def __init__(self, config: Qwen3OmniConfig):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3OmniThinkerForConditionalGeneration(config.thinker_config)
        self.has_talker = config.enable_audio_output
        if config.enable_audio_output:
            self.talker = Qwen3OmniTalkerForConditionalGeneration(config.talker_config)
        else:
            self.talker = None

    def forward(self, *args, **kwargs):
        # TODO: 实现训练时联合 loss / 模式控制（仅文本 / 文本+语音 等）
        return self.thinker(*args, **kwargs)

    # TODO: 可以后续参考 Qwen2_5OmniForConditionalGeneration.generate，
    #       实现 text-only / text+audio 的多模态生成接口
