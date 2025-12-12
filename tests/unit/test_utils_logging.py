from lit_ollama.utils import logging


def test_get_logger() -> None:
    logger = logging.get_logger("test")
    assert logger.name == "test"
    assert logger.level == 10  # DEBUG
