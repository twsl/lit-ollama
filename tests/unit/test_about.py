from lit_ollama import __about__


def test_version_str() -> None:
    assert isinstance(__about__.__version__, str)
