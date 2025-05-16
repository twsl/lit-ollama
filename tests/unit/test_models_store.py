from lit_ollama.models.store import ModelStore


def test_model_store_init():
    store = ModelStore()
    assert hasattr(store, "models")
    assert isinstance(store.models, dict)


def test_model_store_methods(monkeypatch):
    store = ModelStore()
    store.models["foo"] = object()
    store.delete_model("foo")
    assert "foo" not in store.models
