from lit_ollama.models.file import ModelFile


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
