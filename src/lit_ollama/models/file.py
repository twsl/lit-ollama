from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lit_ollama.api.schema.create import CreateRequest

# Known Modelfile parameters and their expected types.
_INT_PARAMS: set[str] = {"num_ctx", "repeat_last_n", "seed", "num_predict", "top_k", "num_keep"}
_FLOAT_PARAMS: set[str] = {
    "repeat_penalty",
    "temperature",
    "top_p",
    "min_p",
    "typical_p",
    "presence_penalty",
    "frequency_penalty",
    "mirostat_tau",
    "mirostat_eta",
}
_MULTI_PARAMS: set[str] = {"stop"}  # parameters that may appear multiple times
VALID_PARAMETERS: set[str] = _INT_PARAMS | _FLOAT_PARAMS | _MULTI_PARAMS | {"mirostat", "penalize_newline"}


def _coerce_param(key: str, value: str) -> Any:
    """Coerce a raw string parameter value to its expected Python type."""
    if key in _INT_PARAMS:
        try:
            return int(value)
        except ValueError:
            return value
    if key in _FLOAT_PARAMS:
        try:
            return float(value)
        except ValueError:
            return value
    if key == "mirostat":
        try:
            return int(value)
        except ValueError:
            return value
    if key == "penalize_newline":
        return value.lower() in ("true", "1", "yes")
    return value


class ModelFile:
    """Build, parse, and convert an Ollama Modelfile.

    Reference: https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx
    """

    def __init__(self) -> None:
        self.base: str | None = None  # FROM (required)
        self.parameters: dict[str, Any] = {}  # values may be lists for multi-value keys
        self.system: str | None = None
        self.template: str | None = None
        self.adapter: str | None = None
        self.license: str | None = None
        self.requires: str | None = None  # minimum Ollama version
        self.messages: list[dict[str, str]] = []
        self.commands: list[str] = []

    # ------------------------------------------------------------------
    # Builder helpers (return ``self`` for chaining)
    # ------------------------------------------------------------------

    def set_base(self, base: str) -> ModelFile:
        self.base = base
        return self

    def set_parameter(self, key: str, value: Any) -> ModelFile:
        """Set a parameter.  Repeated calls for multi-value keys (e.g. ``stop``) append."""
        if key in _MULTI_PARAMS:
            existing = self.parameters.get(key)
            if existing is None:
                self.parameters[key] = [value]
            elif isinstance(existing, list):
                existing.append(value)
            else:
                self.parameters[key] = [existing, value]
        else:
            self.parameters[key] = value
        return self

    def set_system(self, system: str) -> ModelFile:
        self.system = system
        return self

    def set_template(self, template: str) -> ModelFile:
        self.template = template
        return self

    def set_adapter(self, adapter: str) -> ModelFile:
        self.adapter = adapter
        return self

    def set_license(self, license_text: str) -> ModelFile:
        self.license = license_text
        return self

    def set_requires(self, version: str) -> ModelFile:
        self.requires = version
        return self

    def add_message(self, role: str, content: str) -> ModelFile:
        self.messages.append({"role": role, "content": content})
        return self

    def add_command(self, command: str) -> ModelFile:
        self.commands.append(command)
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, *, strict: bool = False) -> list[str]:
        """Validate the Modelfile and return a list of warnings.

        Raises ``ValueError`` if the ``FROM`` instruction is missing.
        When *strict* is ``True``, unknown parameter names also raise ``ValueError``.
        """
        warnings: list[str] = []
        if not self.base:
            raise ValueError("FROM instruction is required")
        for key in self.parameters:
            if key not in VALID_PARAMETERS:
                msg = f"Unknown parameter: {key}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
        return warnings

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_block(keyword: str, value: str) -> str:
        """Render a keyword/value pair, using triple-quote delimiters."""
        return f'{keyword} """{value}"""'

    def render(self) -> str:
        """Serialize the Modelfile to its canonical text representation."""
        lines: list[str] = []

        if self.base:
            lines.append(f"FROM {self.base}")

        for k, v in self.parameters.items():
            if isinstance(v, list):
                for item in v:
                    lines.append(f"PARAMETER {k} {item}")
            else:
                lines.append(f"PARAMETER {k} {v}")

        if self.system:
            lines.append(self._render_block("SYSTEM", self.system))
        if self.template:
            lines.append(self._render_block("TEMPLATE", self.template))
        if self.adapter:
            lines.append(f"ADAPTER {self.adapter}")
        if self.license:
            lines.append(self._render_block("LICENSE", self.license))
        if self.requires:
            lines.append(f"REQUIRES {self.requires}")

        for msg in self.messages:
            content = msg["content"]
            if "\n" in content:
                lines.append(f'MESSAGE {msg["role"]} """{content}"""')
            else:
                lines.append(f"MESSAGE {msg['role']} {content}")

        lines.extend(self.commands)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, text: str) -> ModelFile:
        """Parse a Modelfile from a string."""
        model = cls()
        current_field: str | None = None  # uppercase instruction name
        current_value: list[str] = []
        current_msg_role: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r")
            stripped = line.strip()

            # Skip blanks and comments when not inside a multiline block
            if current_field is None and current_msg_role is None:
                if not stripped or stripped.startswith("#"):
                    continue

            # --- inside a multiline SYSTEM/TEMPLATE/LICENSE block ---
            if current_field is not None:
                if stripped.endswith('"""'):
                    current_value.append(stripped[:-3])
                    setattr(model, current_field.lower(), "\n".join(current_value))
                    current_field = None
                    current_value = []
                else:
                    current_value.append(line if current_value else stripped)
                continue

            # --- inside a multiline MESSAGE block ---
            if current_msg_role is not None:
                if stripped.endswith('"""'):
                    current_value.append(stripped[:-3])
                    model.add_message(current_msg_role, "\n".join(current_value))
                    current_msg_role = None
                    current_value = []
                else:
                    current_value.append(line if current_value else stripped)
                continue

            # --- new instruction (case-insensitive) ---
            upper = stripped.upper()

            # Multiline block start: SYSTEM, TEMPLATE, LICENSE
            ml_match = re.match(r'^(SYSTEM|TEMPLATE|LICENSE)\s+"""(.*)$', stripped, re.IGNORECASE)
            if ml_match:
                current_field = ml_match.group(1).upper()
                rest = ml_match.group(2)
                if rest.endswith('"""'):
                    setattr(model, current_field.lower(), rest[:-3])
                    current_field = None
                else:
                    current_value = [rest]
                continue

            # Single-line string fields (without triple quotes)
            sl_match = re.match(r"^(SYSTEM|TEMPLATE|LICENSE)\s+(.+)$", stripped, re.IGNORECASE)
            if sl_match:
                setattr(model, sl_match.group(1).upper().lower(), sl_match.group(2).strip('"'))
                continue

            # FROM
            if upper.startswith("FROM "):
                model.base = stripped.split(None, 1)[1].strip()
                continue

            # PARAMETER
            if upper.startswith("PARAMETER "):
                parts = stripped.split(None, 2)
                if len(parts) >= 3:
                    key = parts[1]
                    raw_val = parts[2].strip('"')
                    value = _coerce_param(key, raw_val)
                    if key in _MULTI_PARAMS:
                        model.set_parameter(key, value)
                    else:
                        model.parameters[key] = value
                continue

            # ADAPTER
            if upper.startswith("ADAPTER "):
                model.adapter = stripped.split(None, 1)[1].strip()
                continue

            # REQUIRES
            if upper.startswith("REQUIRES "):
                model.requires = stripped.split(None, 1)[1].strip()
                continue

            # MESSAGE (possibly multiline)
            msg_ml = re.match(r'^MESSAGE\s+(\w+)\s+"""(.*)$', stripped, re.IGNORECASE)
            if msg_ml:
                role = msg_ml.group(1)
                rest = msg_ml.group(2)
                if rest.endswith('"""'):
                    model.add_message(role, rest[:-3])
                else:
                    current_msg_role = role
                    current_value = [rest]
                continue

            msg_match = re.match(r"^MESSAGE\s+(\w+)\s+(.+)$", stripped, re.IGNORECASE)
            if msg_match:
                model.add_message(msg_match.group(1), msg_match.group(2).strip('"'))
                continue

            # Arbitrary / unknown instruction
            model.commands.append(stripped)

        return model

    @classmethod
    def from_file(cls, filepath: str) -> ModelFile:
        """Parse a Modelfile from a file path."""
        return cls.from_string(Path(filepath).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Conversion to / from CreateRequest
    # ------------------------------------------------------------------

    def to_create_request(self, model: str = "") -> CreateRequest:
        """Convert this ModelFile into a ``CreateRequest`` schema object."""
        from lit_ollama.api.schema.chat import Message
        from lit_ollama.api.schema.create import CreateRequest

        messages = [Message(role=m["role"], content=m["content"]) for m in self.messages] or None
        return CreateRequest(
            model=model,
            from_=self.base,
            template=self.template,
            license=self.license,
            system=self.system,
            parameters=self.parameters or None,
            messages=messages,
        )

    @classmethod
    def from_create_request(cls, req: CreateRequest) -> ModelFile:
        """Build a ``ModelFile`` from a ``CreateRequest``."""
        mf = cls()
        if req.from_:
            mf.base = req.from_
        if req.template:
            mf.template = req.template
        if req.system:
            mf.system = req.system
        if req.license:
            if isinstance(req.license, list):
                mf.license = "\n".join(req.license)
            else:
                mf.license = req.license
        if req.parameters:
            for k, v in req.parameters.items():
                if isinstance(v, list):
                    for item in v:
                        mf.set_parameter(k, item)
                else:
                    mf.parameters[k] = v
        if req.messages:
            for msg in req.messages:
                role = msg.role if isinstance(msg.role, str) else msg.role.value
                mf.add_message(role, msg.content)
        return mf
