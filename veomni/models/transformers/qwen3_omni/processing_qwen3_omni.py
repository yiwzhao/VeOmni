from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class Qwen3OmniProcessorKwargs:
    # 根据 Qwen2.5 的 Qwen2_5OmniProcessorKwargs 调整
    max_length: Optional[int] = None
    padding: Optional[str] = None
    truncation: Optional[bool] = None
    return_tensors: Optional[str] = None
    # TODO: 补充其他常用 tokenizer kwargs / image/audio kwargs


class Qwen3OmniProcessor(ProcessorMixin):
    """
    Qwen3 Omni Processor: 负责统一处理 text / images / videos / audios。
    类似 Qwen2_5OmniProcessor。
    """

    # TODO: 根据实际使用的类名调整
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "Qwen3VLImageProcessor"  # 或者继续用 Qwen2VLImageProcessor
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen3Tokenizer", "Qwen3TokenizerFast")

    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template: str = None):
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)

        # TODO: 根据 Qwen3 Omni 实际约定的特殊 token 调整
        self.image_token = "<|IMAGE|>"
        self.audio_token = "<|AUDIO|>"
        self.video_token = "<|VIDEO|>"
        self.vision_bos_token = "<|vision_bos|>"
        self.vision_eos_token = "<|vision_eos|>"
        self.audio_bos_token = "<|audio_bos|>"
        self.audio_eos_token = "<|audio_eos|>"

    def __call__(
        self,
        text: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        audios: Optional[List[Any]] = None,
        **kwargs: Qwen3OmniProcessorKwargs,
    ) -> BatchFeature:
        """
        主入口：统一处理输入，返回给模型的 BatchFeature。
        """

        # 1. 处理文本
        texts_inputs: Dict[str, Any] = {}
        if text is not None:
            # TODO: 是否需要先做 multimodal 占位符替换，再 tokenizer？
            texts_inputs = self.tokenizer(
                text,
                padding=kwargs.get("padding", False),
                truncation=kwargs.get("truncation", True),
                max_length=kwargs.get("max_length", None),
                return_tensors=None,
            )

        # 2. 处理图像
        images_inputs: Dict[str, Any] = {}
        image_grid_thw = None
        if images is not None:
            images_inputs = self.image_processor(images=images, return_tensors="pt")
            # TODO: 根据实际 image_processor 接口确定 grid_thw 的字段名字
            image_grid_thw = images_inputs.get("image_grid_thw", None)

        # 3. 处理视频
        videos_inputs: Dict[str, Any] = {}
        video_grid_thw = None
        video_second_per_grid = None
        if videos is not None:
            videos_inputs = self.image_processor(videos=videos, return_tensors="pt")
            # TODO: 确认 Qwen3 Omni 对 video 的处理接口和字段
            video_grid_thw = videos_inputs.get("video_grid_thw", None)
            # TODO: 从 fps / temporal_patch_size 推导 second_per_grid
            video_second_per_grid = videos_inputs.get("video_second_per_grid", None)

        # 4. 处理音频
        audio_inputs: Dict[str, Any] = {}
        audio_lengths = None
        if audios is not None:
            audio_inputs = self.feature_extractor(audios, return_tensors="pt")
            # TODO: 根据 feature_extractor 输出推导 audio_lengths（比如每段的帧数）
            audio_lengths = audio_inputs.get("audio_lengths", None)

        # 5. 替换文本中的 multimodal special tokens（类似 Qwen2.5）
        #    需要 audio_lengths / image_grid_thw / video_grid_thw / video_second_per_grid 等
        # TODO: 实现 replace_multimodal_special_tokens_qwen3
        if text is not None:
            text = self.replace_multimodal_special_tokens_qwen3(
                text,
                audio_lengths=audio_lengths,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                position_id_per_seconds=kwargs.get("position_id_per_seconds", 25),
                seconds_per_chunk=kwargs.get("seconds_per_chunk", 2),
            )
            texts_inputs = self.tokenizer(
                text,
                padding=kwargs.get("padding", False),
                truncation=kwargs.get("truncation", True),
                max_length=kwargs.get("max_length", None),
                return_tensors=None,
            )

        # 6. 汇总为 BatchFeature
        data = {**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs}
        return BatchFeature(data=data, tensor_type=kwargs.get("return_tensors", None))

    def replace_multimodal_special_tokens_qwen3(
        self,
        text: List[str],
        audio_lengths=None,
        image_grid_thw=None,
        video_grid_thw=None,
        video_second_per_grid=None,
        position_id_per_seconds: int = 25,
        seconds_per_chunk: int = 2,
    ) -> List[str]:
        """
        TODO: 参考 Qwen2_5OmniProcessor.replace_multimodal_special_tokens 的逻辑，
        实现 Qwen3 对 <|IMAGE|>, <|AUDIO|>, <|VIDEO|> 等占位符的替换策略。
        """
        # 这里先返回原文，保证 skeleton 可运行，但不做实际替换
        return text
