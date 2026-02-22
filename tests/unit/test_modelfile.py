import pytest

from lit_ollama.server.file import VALID_PARAMETERS, ModelFile


def test_model_file_basic() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.set_parameter("foo", "bar")
    mf.set_system("sys")
    mf.set_template("tmpl")
    mf.set_adapter("adpt")
    mf.set_license("MIT")
    mf.add_message("user", "hi")
    mf.add_command("RUN echo hi")
    rendered = mf.render()
    assert "FROM llama" in rendered
    assert "PARAMETER foo bar" in rendered
    assert "SYSTEM" in rendered
    assert "RUN echo hi" in rendered


# -- from_string: basic -------------------------------------------------------


def test_from_string_basic() -> None:
    text = 'FROM llama3.2\nPARAMETER temperature 0.7\nSYSTEM """You are helpful."""'
    mf = ModelFile.from_string(text)
    assert mf.base == "llama3.2"
    assert mf.parameters["temperature"] == 0.7
    assert mf.system == "You are helpful."


def test_from_string_comments_and_blanks() -> None:
    text = """\n# This is a comment\n\nFROM llama3.2\n\n# Another comment\nPARAMETER seed 42\n"""
    mf = ModelFile.from_string(text)
    assert mf.base == "llama3.2"
    assert mf.parameters["seed"] == 42


def test_from_string_case_insensitive() -> None:
    for keyword in ("from", "From", "FROM"):
        mf = ModelFile.from_string(f"{keyword} llama3.2")
        assert mf.base == "llama3.2", f"Failed for {keyword}"


def test_from_string_parameter_case_insensitive() -> None:
    text = "FROM x\nparameter temperature 0.5"
    mf = ModelFile.from_string(text)
    assert mf.parameters["temperature"] == 0.5


# -- from_string: multiline blocks --------------------------------------------


def test_from_string_multiline_template() -> None:
    text = 'FROM llama\nTEMPLATE """\n{{ .System }}\n{{ .Prompt }}\n"""'
    mf = ModelFile.from_string(text)
    assert "{{ .System }}" in mf.template  # ty:ignore[unsupported-operator]
    assert "{{ .Prompt }}" in mf.template  # ty:ignore[unsupported-operator]


def test_from_string_multiline_system() -> None:
    text = 'FROM llama\nSYSTEM """\nYou are a helpful\nassistant.\n"""'
    mf = ModelFile.from_string(text)
    assert "helpful" in mf.system  # ty:ignore[unsupported-operator]
    assert "assistant" in mf.system  # ty:ignore[unsupported-operator]


def test_from_string_multiline_license() -> None:
    text = 'FROM llama\nLICENSE """\nMIT License\nCopyright 2025\n"""'
    mf = ModelFile.from_string(text)
    assert "MIT License" in mf.license  # ty:ignore[unsupported-operator]
    assert "Copyright 2025" in mf.license  # ty:ignore[unsupported-operator]


def test_from_string_multiline_message() -> None:
    text = 'FROM llama\nMESSAGE user """\nHello\nWorld\n"""'
    mf = ModelFile.from_string(text)
    assert len(mf.messages) == 1
    assert mf.messages[0]["role"] == "user"
    assert "Hello" in mf.messages[0]["content"]
    assert "World" in mf.messages[0]["content"]


# -- from_string: parameter handling ------------------------------------------


def test_from_string_type_coercion() -> None:
    text = (
        "FROM llama\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER seed 42\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER min_p 0.05\n"
        "PARAMETER repeat_last_n 64\n"
        "PARAMETER repeat_penalty 1.1\n"
        "PARAMETER num_predict 128\n"
        "PARAMETER top_k 40\n"
    )
    mf = ModelFile.from_string(text)
    assert mf.parameters["num_ctx"] == 4096
    assert isinstance(mf.parameters["num_ctx"], int)
    assert mf.parameters["temperature"] == 0.7
    assert isinstance(mf.parameters["temperature"], float)
    assert mf.parameters["seed"] == 42
    assert isinstance(mf.parameters["seed"], int)
    assert mf.parameters["top_p"] == 0.9
    assert isinstance(mf.parameters["top_p"], float)
    assert mf.parameters["min_p"] == 0.05
    assert isinstance(mf.parameters["min_p"], float)
    assert mf.parameters["repeat_last_n"] == 64
    assert isinstance(mf.parameters["repeat_last_n"], int)
    assert mf.parameters["repeat_penalty"] == 1.1
    assert isinstance(mf.parameters["repeat_penalty"], float)
    assert mf.parameters["num_predict"] == 128
    assert isinstance(mf.parameters["num_predict"], int)
    assert mf.parameters["top_k"] == 40
    assert isinstance(mf.parameters["top_k"], int)


def test_from_string_multiple_stop() -> None:
    text = (
        "FROM llama\n"
        'PARAMETER stop "<|start_header_id|>"\n'
        'PARAMETER stop "<|end_header_id|>"\n'
        'PARAMETER stop "<|eot_id|>"\n'
    )
    mf = ModelFile.from_string(text)
    stops = mf.parameters["stop"]
    assert isinstance(stops, list)
    assert len(stops) == 3
    assert "<|start_header_id|>" in stops
    assert "<|end_header_id|>" in stops
    assert "<|eot_id|>" in stops


# -- from_string: REQUIRES ----------------------------------------------------


def test_from_string_requires() -> None:
    text = "FROM llama\nREQUIRES 0.14.0"
    mf = ModelFile.from_string(text)
    assert mf.requires == "0.14.0"


def test_from_string_requires_case_insensitive() -> None:
    text = "FROM llama\nrequires 0.14.0"
    mf = ModelFile.from_string(text)
    assert mf.requires == "0.14.0"


# -- from_string: ADAPTER -----------------------------------------------------


def test_from_string_adapter() -> None:
    text = "FROM llama\nADAPTER ./my-lora.gguf"
    mf = ModelFile.from_string(text)
    assert mf.adapter == "./my-lora.gguf"


# -- from_string: messages ----------------------------------------------------


def test_from_string_single_line_messages() -> None:
    text = (
        "FROM llama\n"
        "MESSAGE user Is Toronto in Canada?\n"
        "MESSAGE assistant yes\n"
        "MESSAGE user Is Sacramento in Canada?\n"
        "MESSAGE assistant no\n"
    )
    mf = ModelFile.from_string(text)
    assert len(mf.messages) == 4
    assert mf.messages[0] == {"role": "user", "content": "Is Toronto in Canada?"}
    assert mf.messages[1] == {"role": "assistant", "content": "yes"}


# -- render -------------------------------------------------------------------


def test_render_requires() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.set_requires("0.14.0")
    rendered = mf.render()
    assert "REQUIRES 0.14.0" in rendered


def test_render_multiple_stop() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.set_parameter("stop", "<|start|>")
    mf.set_parameter("stop", "<|end|>")
    rendered = mf.render()
    assert rendered.count("PARAMETER stop") == 2
    assert "<|start|>" in rendered
    assert "<|end|>" in rendered


def test_render_multiline_message() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.add_message("user", "Hello\nWorld")
    rendered = mf.render()
    assert '"""Hello\nWorld"""' in rendered


def test_render_single_line_message() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.add_message("user", "Hello")
    rendered = mf.render()
    assert "MESSAGE user Hello" in rendered
    assert '"""' not in rendered.split("MESSAGE")[1]


# -- round-trip ---------------------------------------------------------------


def test_render_roundtrip() -> None:
    original = (
        "FROM llama3.2\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER stop <|start|>\n"
        "PARAMETER stop <|end|>\n"
        'SYSTEM """You are helpful."""\n'
        'TEMPLATE """{{ .Prompt }}"""\n'
        "ADAPTER ./lora.gguf\n"
        'LICENSE """MIT"""\n'
        "REQUIRES 0.14.0\n"
        "MESSAGE user Hello\n"
        "MESSAGE assistant Hi there"
    )
    mf = ModelFile.from_string(original)
    rendered = mf.render()
    mf2 = ModelFile.from_string(rendered)
    assert mf2.base == mf.base
    assert mf2.parameters == mf.parameters
    assert mf2.system == mf.system
    assert mf2.template == mf.template
    assert mf2.adapter == mf.adapter
    assert mf2.license == mf.license
    assert mf2.requires == mf.requires
    assert mf2.messages == mf.messages


# -- validation ---------------------------------------------------------------


def test_validate_missing_from() -> None:
    mf = ModelFile()
    with pytest.raises(ValueError, match="FROM instruction is required"):
        mf.validate()


def test_validate_unknown_parameter_warning() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.parameters["unknown_param"] = 42
    warnings = mf.validate()
    assert len(warnings) == 1
    assert "unknown_param" in warnings[0]


def test_validate_unknown_parameter_strict() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.parameters["unknown_param"] = 42
    with pytest.raises(ValueError, match="Unknown parameter"):
        mf.validate(strict=True)


def test_validate_valid_model() -> None:
    mf = ModelFile()
    mf.set_base("llama")
    mf.set_parameter("temperature", 0.7)
    mf.set_parameter("stop", "<|end|>")
    warnings = mf.validate()
    assert warnings == []


# -- to_create_request / from_create_request ----------------------------------


def test_to_create_request() -> None:
    mf = ModelFile()
    mf.set_base("llama3.2")
    mf.set_system("You are helpful.")
    mf.set_template("{{ .Prompt }}")
    mf.set_license("MIT")
    mf.parameters["temperature"] = 0.7
    mf.add_message("user", "Hello")

    req = mf.to_create_request(model="my-model")
    assert req.model == "my-model"
    assert req.from_ == "llama3.2"
    assert req.system == "You are helpful."
    assert req.template == "{{ .Prompt }}"
    assert req.license == "MIT"
    assert req.parameters["temperature"] == 0.7  # ty:ignore[not-subscriptable]
    assert len(req.messages) == 1  # ty:ignore[invalid-argument-type]
    assert req.messages[0].content == "Hello"  # ty:ignore[not-subscriptable]


def test_from_create_request() -> None:
    from lit_ollama.server.schema.chat import Message
    from lit_ollama.server.schema.create import CreateRequest

    req = CreateRequest(
        model="test",
        from_="llama3.2",
        system="Be helpful",
        template="{{ .Prompt }}",
        license="MIT",
        parameters={"temperature": 0.7, "stop": ["<|end|>", "<|start|>"]},
        messages=[Message(role="user", content="Hi")],
    )
    mf = ModelFile.from_create_request(req)
    assert mf.base == "llama3.2"
    assert mf.system == "Be helpful"
    assert mf.template == "{{ .Prompt }}"
    assert mf.license == "MIT"
    assert mf.parameters["temperature"] == 0.7
    assert mf.parameters["stop"] == ["<|end|>", "<|start|>"]
    assert len(mf.messages) == 1
    assert mf.messages[0]["content"] == "Hi"


def test_from_create_request_license_list() -> None:
    from lit_ollama.server.schema.create import CreateRequest

    req = CreateRequest(model="test", license=["MIT", "Apache-2.0"])
    mf = ModelFile.from_create_request(req)
    assert mf.license == "MIT\nApache-2.0"


# -- builder chaining ---------------------------------------------------------


def test_builder_chaining() -> None:
    mf = (
        ModelFile()
        .set_base("llama3.2")
        .set_parameter("temperature", 0.7)
        .set_system("Be helpful")
        .set_template("{{ .Prompt }}")
        .set_adapter("./lora.gguf")
        .set_license("MIT")
        .set_requires("0.14.0")
        .add_message("user", "Hello")
        .add_command("RUN echo hi")
    )
    assert mf.base == "llama3.2"
    assert mf.requires == "0.14.0"
    assert len(mf.messages) == 1


# -- comprehensive examples from ollama docs ----------------------------------


def test_ollama_docs_example() -> None:
    """Parse the Mario example from Ollama's Modelfile documentation."""
    text = """FROM llama3.2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096
PARAMETER num_ctx 4096

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
"""
    mf = ModelFile.from_string(text)
    assert mf.base == "llama3.2"
    assert mf.parameters["temperature"] == 1.0
    assert mf.parameters["num_ctx"] == 4096
    assert "Mario" in mf.system  # ty:ignore[unsupported-operator]
    warnings = mf.validate()
    assert warnings == []


def test_ollama_show_modelfile_example() -> None:
    """Parse the output of ``ollama show --modelfile``."""
    text = '''# Modelfile generated by "ollama show"
# To build a new Modelfile based on this one, replace the FROM line with:
# FROM llama3.2:latest
FROM /Users/pdevine/.ollama/models/blobs/sha256-00e1317cbf74d901
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
'''
    mf = ModelFile.from_string(text)
    assert mf.base == "/Users/pdevine/.ollama/models/blobs/sha256-00e1317cbf74d901"
    assert "{{ .System }}" in mf.template  # ty:ignore[unsupported-operator]
    assert "{{ .Prompt }}" in mf.template  # ty:ignore[unsupported-operator]
    stops = mf.parameters["stop"]
    assert isinstance(stops, list)
    assert len(stops) == 4


# -- VALID_PARAMETERS constant ------------------------------------------------


def test_valid_parameters_contains_all_documented_params() -> None:
    documented = {
        "num_ctx",
        "repeat_last_n",
        "repeat_penalty",
        "temperature",
        "seed",
        "stop",
        "num_predict",
        "top_k",
        "top_p",
        "min_p",
    }
    assert documented.issubset(VALID_PARAMETERS)
