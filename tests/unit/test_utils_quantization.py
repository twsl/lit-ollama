from lit_ollama.server.schema.base import QuantizationType
from lit_ollama.utils.quantization import quantize_mapping


def test_quantize_mapping_keys() -> None:
    for key in QuantizationType:
        assert key in quantize_mapping
