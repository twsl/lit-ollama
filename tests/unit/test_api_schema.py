from datetime import datetime

from lit_ollama.api.schema.base import ModelDetails, RunningModel, TagModel
from lit_ollama.api.schema.chat import ChatResponse, Message
from lit_ollama.api.schema.copy import CopyRequest
from lit_ollama.api.schema.create import CreateRequest, CreateResponse
from lit_ollama.api.schema.custom.ls import LsResponse
from lit_ollama.api.schema.delete import DeleteRequest
from lit_ollama.api.schema.embed import EmbedRequest, EmbedResponse
from lit_ollama.api.schema.generate import GenerateRequest, GenerateResponse
from lit_ollama.api.schema.ps import PsResponse
from lit_ollama.api.schema.pull import PullRequest, PullResponse
from lit_ollama.api.schema.push import PushRequest, PushResponse
from lit_ollama.api.schema.show import ShowRequest, ShowResponse
from lit_ollama.api.schema.tags import TagsResponse


def test_model_details() -> None:
    d = ModelDetails("gguf", "llama", ["llama"], "1B", "q4_0")
    assert d.format == "gguf"


def test_tag_model() -> None:
    d = ModelDetails("gguf", "llama", ["llama"], "1B", "q4_0")
    t = TagModel("foo", datetime.now(), 123, "digest", d)
    assert t.name == "foo"


def test_chat_response_serialize() -> None:
    m = Message("user", "hi")
    c = ChatResponse("m", datetime.now(), m)
    data = c.serialize()
    assert "created_at" in data


def test_copy_request() -> None:
    c = CopyRequest("a", "b")
    assert c.source == "a"


def test_create_response() -> None:
    c = CreateResponse("ok")
    assert c.status == "ok"


def test_ls_response() -> None:
    l = LsResponse(["foo"])
    assert l.models == ["foo"]


def test_delete_request() -> None:
    d = DeleteRequest("foo")
    assert d.model == "foo"


def test_embed_response() -> None:
    e = EmbedResponse("m", [[0.1, 0.2]], 1, 2, 3)
    assert e.model == "m"


def test_generate_response() -> None:
    g = GenerateResponse("m", datetime.now(), "hi")
    assert g.response == "hi"


def test_ps_response() -> None:
    p = PsResponse([])
    assert isinstance(p.models, list)


def test_pull_response() -> None:
    p = PullResponse("ok")
    assert p.status == "ok"


def test_push_response() -> None:
    p = PushResponse("ok")
    assert p.status == "ok"


def test_show_response() -> None:
    s = ShowResponse("modelfile", "params", "tmpl", {}, {})
    assert s.modelfile == "modelfile"


def test_tags_response() -> None:
    t = TagsResponse([])
    assert isinstance(t.models, list)
