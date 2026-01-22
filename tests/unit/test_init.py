def test_import_version():
    from lit_ollama import __version__

    assert isinstance(__version__, str)
