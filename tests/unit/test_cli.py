from lit_ollama import cli


def test_main_prints_version(capsys):
    cli.main()
    out, _ = capsys.readouterr()
    assert "lit-ollama v" in out
