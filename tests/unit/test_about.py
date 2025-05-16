from lit_ollama import __about__


def test_version_tuple():
    assert isinstance(__about__.__version_tuple__, tuple)
    assert len(__about__.__version_tuple__) == 3


def test_version_str():
    assert isinstance(__about__.__version__, str)
