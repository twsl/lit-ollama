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


def test_model_details():
    d = ModelDetails("gguf", "llama", ["llama"], "1B", "q4_0")
    assert d.format == "gguf"


def test_tag_model():
    d = ModelDetails("gguf", "llama", ["llama"], "1B", "q4_0")
    t = TagModel("foo", datetime.now(), 123, "digest", d)
    assert t.name == "foo"


def test_chat_response_serialize():
    m = Message("user", "hi")
    c = ChatResponse("m", datetime.now(), m)
    data = c.serialize()
    assert "created_at" in data


def test_copy_request():
    c = CopyRequest("a", "b")
    assert c.source == "a"


def test_create_response():
    c = CreateResponse("ok")
    assert c.status == "ok"


def test_ls_response():
    l = LsResponse(["foo"])
    assert l.models == ["foo"]


def test_delete_request():
    d = DeleteRequest("foo")
    assert d.model == "foo"


def test_embed_response():
    e = EmbedResponse("m", [[0.1, 0.2]], 1, 2, 3)
    assert e.model == "m"


def test_generate_response():
    g = GenerateResponse("m", datetime.now(), "hi")
    assert g.response == "hi"


def test_ps_response():
    p = PsResponse([])
    assert isinstance(p.models, list)


def test_pull_response():
    p = PullResponse("ok")
    assert p.status == "ok"


def test_push_response():
    p = PushResponse("ok")
    assert p.status == "ok"


def test_show_response():
    s = ShowResponse("modelfile", "params", "tmpl", {}, {})
    assert s.modelfile == "modelfile"


def test_tags_response():
    t = TagsResponse([])
    assert isinstance(t.models, list)
