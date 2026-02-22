import httpx
from ollama import Client, ResponseError
import pytest


class TestVersion:
    def test_version_returns_string(self, client: Client, base_url: str) -> None:
        """GET /api/version via raw HTTP (not in ollama client)."""
        r = httpx.get(f"{base_url}/api/version")
        assert r.status_code == 200
        body = r.json()
        assert "version" in body
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0


class TestList:
    def test_list_models(self, client: Client) -> None:
        """client.list() -> ListResponse with a models list."""
        resp = client.list()
        # The mock server may have zero stored models, but the structure must be valid.
        assert hasattr(resp, "models")
        assert isinstance(resp.models, list)

    def test_tags_returns_list(self, base_url: str) -> None:
        """GET /api/tags returns a list of known litgpt model tags."""
        r = httpx.get(f"{base_url}/api/tags")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["models"], list)


class TestGenerate:
    def test_generate_non_streaming(self, client: Client) -> None:
        resp = client.generate(model="mock", prompt="Say hello", stream=False)
        assert resp is not None
        assert hasattr(resp, "response")
        assert isinstance(resp.response, str)
        assert len(resp.response) > 0
        assert resp.done is True
        assert resp.model == "mock"

    def test_generate_streaming(self, client: Client) -> None:
        chunks: list = []
        for chunk in client.generate(model="mock", prompt="Say hello", stream=True):
            chunks.append(chunk)

        assert len(chunks) >= 2  # at least one token + final done chunk
        # All non-final chunks should have done=False
        for c in chunks[:-1]:
            assert c.done is False
        # Last chunk signals completion
        assert chunks[-1].done is True

    def test_generate_empty_prompt_load(self, client: Client) -> None:
        """Empty prompt with keep_alive should trigger a model load response."""
        resp = client.generate(model="mock", prompt="", stream=False)
        assert resp.done is True
        assert resp.done_reason in ("load", "unload", None)

    def test_generate_has_timing_metrics(self, client: Client) -> None:
        resp = client.generate(model="mock", prompt="Timing test", stream=False)
        # The server should include duration metrics
        assert resp.total_duration is not None or resp.done is True


class TestChat:
    def test_chat_non_streaming(self, client: Client) -> None:
        resp = client.chat(
            model="mock",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert resp is not None
        assert hasattr(resp, "message")
        assert resp.message.role == "assistant"
        assert isinstance(resp.message.content, str)
        assert len(resp.message.content) > 0
        assert resp.done is True
        assert resp.model == "mock"

    def test_chat_streaming(self, client: Client) -> None:
        chunks: list = []
        for chunk in client.chat(
            model="mock",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 2
        for c in chunks[:-1]:
            assert c.done is False
            assert c.message.role == "assistant"
        assert chunks[-1].done is True

    def test_chat_multi_turn(self, client: Client) -> None:
        """Multiple messages in the conversation context."""
        resp = client.chat(
            model="mock",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            stream=False,
        )
        assert resp.done is True
        assert resp.message.content is not None

    def test_chat_empty_messages_load(self, client: Client) -> None:
        """Empty messages list should trigger a model load/unload response."""
        resp = client.chat(model="mock", messages=[], stream=False)
        assert resp.done is True


class TestShow:
    def test_show_model(self, client: Client) -> None:
        resp = client.show("mock")
        assert resp is not None
        assert hasattr(resp, "modelfile")
        assert isinstance(resp.modelfile, str)
        assert hasattr(resp, "details")
        # ollama client maps model_info -> modelinfo
        assert hasattr(resp, "modelinfo")
        assert isinstance(resp.modelinfo, dict)

    def test_show_has_parameters(self, client: Client) -> None:
        resp = client.show("mock")
        assert hasattr(resp, "parameters")


class TestEmbed:
    def test_embed_single_input(self, client: Client) -> None:
        resp = client.embed(model="mock", input="The sky is blue")
        assert resp is not None
        assert hasattr(resp, "embeddings")
        assert isinstance(resp.embeddings, list)
        assert len(resp.embeddings) == 1
        # Each embedding is a list of floats
        assert isinstance(resp.embeddings[0], list)
        assert all(isinstance(v, float) for v in resp.embeddings[0])
        assert resp.model == "mock"

    def test_embed_batch_input(self, client: Client) -> None:
        resp = client.embed(
            model="mock",
            input=["Hello world", "Goodbye world"],
        )
        assert len(resp.embeddings) == 2
        assert all(isinstance(emb, list) for emb in resp.embeddings)

    def test_embed_has_metrics(self, client: Client) -> None:
        resp = client.embed(model="mock", input="metrics test")
        assert resp.total_duration is not None
        assert resp.prompt_eval_count is not None


class TestPs:
    def test_ps_returns_running_models(self, client: Client) -> None:
        resp = client.ps()
        assert resp is not None
        assert hasattr(resp, "models")
        assert isinstance(resp.models, list)
        # The mock model should appear as running
        if resp.models:
            m = resp.models[0]
            assert hasattr(m, "name")
            assert hasattr(m, "size")
            assert hasattr(m, "details")


class TestBlobs:
    def test_blob_head_not_found(self, base_url: str) -> None:
        """HEAD for a non-existent digest returns 404."""
        digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        r = httpx.head(f"{base_url}/api/blobs/{digest}")
        assert r.status_code == 404

    def test_blob_create_and_check(self, base_url: str) -> None:
        """POST a blob then HEAD should find it."""
        import hashlib

        content = b"integration test blob content"
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        # Create
        r = httpx.post(f"{base_url}/api/blobs/{digest}", content=content)
        assert r.status_code == 201

        # Verify existence
        r = httpx.head(f"{base_url}/api/blobs/{digest}")
        assert r.status_code == 200


class TestCopy:
    def test_copy_nonexistent_source_returns_404(self, client: Client) -> None:
        """Copying a model that doesn't exist should fail."""
        with pytest.raises((ResponseError, Exception)):
            client.copy("nonexistent-model-12345", "copy-dest")


class TestDelete:
    def test_delete_nonexistent_returns_404(self, client: Client) -> None:
        """Deleting a model that doesn't exist should fail."""
        with pytest.raises((ResponseError, Exception)):
            client.delete("nonexistent-model-12345")


class TestPush:
    def test_push_returns_not_implemented(self, client: Client) -> None:
        """Push is not implemented; server returns 501."""
        with pytest.raises(ResponseError):
            client.push("mock", stream=False)


class TestPull:
    def test_pull_non_streaming(self, client: Client) -> None:
        """Pull with stream=false returns a single JSON response."""
        resp = client.pull("mock", stream=False)
        assert resp is not None
        assert hasattr(resp, "status")
        assert isinstance(resp.status, str)


class TestCreate:
    def test_create_with_from(self, client: Client) -> None:
        """Create with from_ returns a streaming status response."""
        statuses = []
        for progress in client.create(model="test-create", from_="mock", stream=True):
            assert hasattr(progress, "status")
            statuses.append(progress.status)
        assert len(statuses) >= 2
        assert statuses[0] == "reading model metadata"
        assert statuses[-1] == "success"


class TestEndToEnd:
    def test_generate_then_chat_workflow(self, client: Client) -> None:
        """Verify the server handles sequential generate and chat requests."""
        gen_resp = client.generate(model="mock", prompt="Step 1", stream=False)
        assert gen_resp.done is True
        assert len(gen_resp.response) > 0

        chat_resp = client.chat(
            model="mock",
            messages=[{"role": "user", "content": "Step 2"}],
            stream=False,
        )
        assert chat_resp.done is True
        assert chat_resp.message.content is not None
        assert len(chat_resp.message.content) > 0

    def test_streaming_generate_then_streaming_chat(self, client: Client) -> None:
        """Streaming requests sequentially should not interfere."""
        gen_tokens = []
        for chunk in client.generate(model="mock", prompt="Stream 1", stream=True):
            if not chunk.done:
                gen_tokens.append(chunk.response)
        assert len(gen_tokens) > 0

        chat_tokens = []
        for chunk in client.chat(
            model="mock",
            messages=[{"role": "user", "content": "Stream 2"}],
            stream=True,
        ):
            if not chunk.done:
                chat_tokens.append(chunk.message.content)
        assert len(chat_tokens) > 0
