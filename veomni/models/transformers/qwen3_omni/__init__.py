from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

from transformers.utils import logging
from ..utils.import_utils import is_transformers_version_greater_or_equal_to  # 路径按实际工程调整

from .configuration_qwen3_omni import Qwen3OmniConfig
from .modeling_qwen3_omni import Qwen3OmniForConditionalGeneration
from .processing_qwen3_omni import Qwen3OmniProcessor

logger = logging.get_logger(__name__)

# TODO: 确认从哪个 transformers 版本开始官方支持 Qwen3 Omni，如果没有官方支持，这个判断可以先去掉
if not is_transformers_version_greater_or_equal_to("4.99.0"):
    AutoConfig.register("qwen3_omni", Qwen3OmniConfig)
    AutoModel.register(Qwen3OmniConfig, Qwen3OmniForConditionalGeneration)
    AutoModelForCausalLM.register(Qwen3OmniConfig, Qwen3OmniForConditionalGeneration)
    AutoProcessor.register(Qwen3OmniConfig, Qwen3OmniProcessor)
