def test_import_version():
    from lit_ollama import __version__, __version_tuple__

    assert isinstance(__version__, str)
    assert isinstance(__version_tuple__, tuple)
