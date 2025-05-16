from lit_ollama.api.schema.base import QuantizationType
from lit_ollama.utils.quantization import quantize_mapping


def test_quantize_mapping_keys():
    for key in QuantizationType:
        assert key in quantize_mapping
