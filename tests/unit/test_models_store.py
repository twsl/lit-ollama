from lit_ollama.store.models import ModelStore


def test_model_store_init() -> None:
    store = ModelStore()
    assert hasattr(store, "models")
    assert isinstance(store.models, dict)


def test_model_store_methods() -> None:
    store = ModelStore()
    store.models["foo"] = object()  # pyright: ignore[reportArgumentType]
    store.delete_model("foo")
    assert "foo" not in store.models
