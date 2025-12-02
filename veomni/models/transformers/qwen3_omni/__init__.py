from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from .configuration_qwen3_omni import Qwen3OmniConfig
from .modeling_qwen3_omni import Qwen3OmniForConditionalGeneration
from .processing_qwen3_omni import Qwen3OmniProcessor


# After 4.57, this model is already registered in transformers. Registering again
# will cause "already exists" errors.
if not is_transformers_version_greater_or_equal_to("4.57.0"):
    AutoConfig.register("qwen3_omni", Qwen3OmniConfig)
    AutoModel.register(Qwen3OmniConfig, Qwen3OmniForConditionalGeneration)
    AutoModelForCausalLM.register(Qwen3OmniConfig, Qwen3OmniForConditionalGeneration)
    AutoProcessor.register(Qwen3OmniConfig, Qwen3OmniProcessor)






# After 4.52, this model is already registered in transfomers. Register will cause
# already exists error.
if not is_transformers_version_greater_or_equal_to("4.52.0"):
    AutoConfig.register("qwen2_5_omni", Qwen2_5OmniConfig)
    AutoModel.register(Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration)
    AutoModelForCausalLM.register(Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration)
    AutoProcessor.register(Qwen2_5OmniConfig, Qwen2_5OmniProcessor)
