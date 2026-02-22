"""Utilities to extract real model metadata from litgpt objects.

The Ollama-compatible API responses (e.g. /api/show, /api/tags, /api/ps) need
information that is available on litgpt's ``LLM`` object and its ``Config``.

This module is intentionally defensive:
- It supports both litgpt Config objects and dict-like configs.
- It supports mock/dev LLM implementations that may omit attributes.
"""

from __future__ import annotations

from typing import Any

from lit_ollama.server.schema.base import ModelDetails, ShowModelDetails


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_hf_config(config: Any) -> dict[str, Any]:
    hf = _get(config, "hf_config", {})
    return hf if isinstance(hf, dict) else {}


def _derive_family(config: Any) -> str:
    hf = _get_hf_config(config)
    org = str(hf.get("org", "") or "").strip()

    org_map: dict[str, str] = {
        "meta-llama": "llama",
        "google": "gemma",
        "mistralai": "mistral",
        "microsoft": "phi",
        "qwen": "qwen",
    }
    if org in org_map:
        return org_map[org]

    name = str(_get(config, "name", "") or "").lower()
    if "llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name:
        return "mistral"
    if "phi" in name:
        return "phi"
    if "qwen" in name:
        return "qwen"

    return org or "unknown"


def _families(config: Any) -> list[str]:
    fam = _derive_family(config)
    return [] if fam == "unknown" else [fam]


def _format_parameter_size(parameter_count: int) -> str:
    if parameter_count <= 0:
        return ""
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.1f}B"
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.0f}M"
    if parameter_count >= 1_000:
        return f"{parameter_count / 1_000:.0f}K"
    return str(parameter_count)


def estimate_parameter_count(config: Any) -> int:
    """Cheap estimate from config dimensions.

    Used when the full torch model isn't loaded (e.g., when scanning checkpoints).
    """
    n_embd = int(_get(config, "n_embd", 0) or 0)
    n_layer = int(_get(config, "n_layer", 0) or 0)
    vocab_size = int(_get(config, "vocab_size", 0) or 0)
    intermediate_size = int(_get(config, "intermediate_size", 0) or 0) or (4 * n_embd)

    if n_embd <= 0 or n_layer <= 0:
        return 0

    # Very rough Transformer parameter estimate.
    embed = vocab_size * n_embd
    attn = 4 * n_embd * n_embd
    ffn = 3 * n_embd * intermediate_size
    norms = 4 * n_embd
    block = attn + ffn + norms
    lm_head = vocab_size * n_embd

    return embed + n_layer * block + lm_head


def count_model_parameters(llm: Any) -> int:
    model = getattr(llm, "model", None)
    if model is None:
        return 0
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return 0


def build_model_details(config: Any, *, parameter_count: int = 0, quantization_level: str = "") -> ModelDetails:
    fam = _derive_family(config)
    if not parameter_count:
        parameter_count = estimate_parameter_count(config)
    return ModelDetails(
        format="pytorch",
        family=fam,
        families=_families(config),
        parameter_size=_format_parameter_size(parameter_count),
        quantization_level=quantization_level,
        parent_model="",
    )


def build_show_model_details(
    config: Any, *, parameter_count: int = 0, quantization_level: str = ""
) -> ShowModelDetails:
    fam = _derive_family(config)
    if not parameter_count:
        parameter_count = estimate_parameter_count(config)
    return ShowModelDetails(
        format="pytorch",
        family=fam,
        families=_families(config),
        parameter_size=_format_parameter_size(parameter_count),
        quantization_level=quantization_level,
        parent_model="",
    )


def build_model_info(config: Any, *, llm: Any | None = None, verbose: bool = False) -> dict[str, Any]:
    """Build an Ollama-style model_info dict."""
    arch = _derive_family(config)
    if arch == "unknown":
        arch = "llama"

    parameter_count = count_model_parameters(llm) if llm is not None else estimate_parameter_count(config)

    info: dict[str, Any] = {
        "general.architecture": arch,
        "general.parameter_count": parameter_count,
        f"{arch}.attention.head_count": _get(config, "n_head", None),
        f"{arch}.attention.head_count_kv": _get(config, "n_query_groups", None),
        f"{arch}.attention.layer_norm_rms_epsilon": _get(config, "norm_eps", None),
        f"{arch}.block_count": _get(config, "n_layer", None),
        f"{arch}.context_length": _get(config, "block_size", None),
        f"{arch}.embedding_length": _get(config, "n_embd", None),
        f"{arch}.feed_forward_length": _get(config, "intermediate_size", None),
        f"{arch}.rope.dimension_count": _get(config, "rope_n_elem", None),
        f"{arch}.rope.freq_base": _get(config, "rope_base", None),
        f"{arch}.vocab_size": _get(config, "vocab_size", None),
    }

    if llm is not None:
        tokenizer = getattr(llm, "tokenizer", None)
        if tokenizer is not None:
            info["tokenizer.ggml.bos_token_id"] = getattr(tokenizer, "bos_id", None)
            info["tokenizer.ggml.eos_token_id"] = getattr(tokenizer, "eos_id", None)
            backend = getattr(tokenizer, "backend", None)
            if backend is not None:
                info["tokenizer.ggml.model"] = backend

            if verbose:
                # Only attempt if tokenizer has these attributes; many backends won't.
                for key, attr in (
                    ("tokenizer.ggml.tokens", "tokens"),
                    ("tokenizer.ggml.merges", "merges"),
                    ("tokenizer.ggml.token_type", "token_type"),
                    ("tokenizer.ggml.pre", "pre"),
                ):
                    if hasattr(tokenizer, attr):
                        info[key] = getattr(tokenizer, attr)

    return {k: v for k, v in info.items() if v is not None}


def template_name(llm: Any) -> str:
    prompt_style = getattr(llm, "prompt_style", None)
    return type(prompt_style).__name__ if prompt_style is not None else ""


def capabilities(llm: Any) -> list[str]:
    caps = ["completion"]
    model = getattr(llm, "model", None)
    transformer = getattr(model, "transformer", None) if model is not None else None
    if transformer is not None and hasattr(transformer, "wte"):
        caps.append("embedding")
    return caps


def build_parameters_string(config: Any, *, llm: Any | None = None) -> str:
    """Build the text block for ShowResponse.parameters.

    Ollama's `show` command prints one setting per line. We only populate the
    fields we can derive reliably from litgpt.
    """
    lines: list[str] = []
    num_ctx = _get(config, "block_size", None)
    if num_ctx is not None:
        lines.append(f"num_ctx                        {num_ctx}")

    if llm is not None:
        tok = getattr(llm, "tokenizer", None)
        if tok is not None:
            eos_id = getattr(tok, "eos_id", None)
            if eos_id is not None:
                # We don't know the literal token string; keep a stable placeholder.
                lines.append(f"stop                           <|eos_token_id:{eos_id}|>")

    return "\n".join(lines)
