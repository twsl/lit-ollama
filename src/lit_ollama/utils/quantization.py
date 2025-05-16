from lit_ollama.api.schema.base import QuantizationType

# mapping doesnt align at all, but exists to provide quantization option

quantize_mapping = {
    # https://github.com/ollama/ollama/blob/main/docs/import.md#supported-quantizations
    QuantizationType.q4_0: "bnb.fp4",
    QuantizationType.q4_1: "bnb.fp4-dq",
    QuantizationType.q5_0: "bnb.nf4",
    QuantizationType.q5_1: "bnb.nf4-dq",
    QuantizationType.q8_0: "bnb.int8",
    # https://github.com/ollama/ollama/blob/main/docs/import.md#k-means-quantizations
    QuantizationType.q2_K: None,
    QuantizationType.q3_K_L: None,
    QuantizationType.q3_K_M: None,
    QuantizationType.q3_K_S: None,
    QuantizationType.q4_K_M: None,
    QuantizationType.q4_K_S: None,
    QuantizationType.q5_K_M: None,
    QuantizationType.q5_K_S: None,
    QuantizationType.q6_K: None,
}
