from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import ShowModelDetails

# https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information


@dataclass
class ShowRequest:
    model: str
    verbose: bool | None = None


# @dataclass
# class ShowModelInfo:
#     general.architecture": "llama",
#     general.file_type": 2,
#     general.parameter_count": 8030261248,
#     general.quantization_version": 2,
#     llama.attention.head_count": 32,
#     llama.attention.head_count_kv": 8,
#     llama.attention.layer_norm_rms_epsilon": 0.00001,
#     llama.block_count": 32,
#     llama.context_length": 8192,
#     llama.embedding_length": 4096,
#     llama.feed_forward_length": 14336,
#     llama.rope.dimension_count": 128,
#     llama.rope.freq_base": 500000,
#     llama.vocab_size": 128256,
#     tokenizer.ggml.bos_token_id": 128000,
#     tokenizer.ggml.eos_token_id": 128009,
#     tokenizer.ggml.merges": [],            // populates if `verbose=true`
#     tokenizer.ggml.model": "gpt2",
#     tokenizer.ggml.pre": "llama-bpe",
#     tokenizer.ggml.token_type": [],        // populates if `verbose=true`
#     tokenizer.ggml.tokens": []             // populates if `verbose=true`


@dataclass
class ShowResponse:
    modelfile: str
    parameters: str
    template: str
    details: ShowModelDetails | dict
    model_info: dict
