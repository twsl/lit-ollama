from pathlib import Path
from typing import Any


class ModelFile:
    """Generate a Modelfile based on https://github.com/ollama/ollama/blob/main/docs/modelfile.md."""

    def __init__(self) -> None:
        self.base: str | None = None  # FROM
        self.parameters: dict[str, Any] = {}
        self.system: str | None = None
        self.template: str | None = None
        self.adapter: str | None = None
        self.license: str | None = None
        self.messages: list[dict[str, str]] = []
        self.commands: list[str] = []

    def set_base(self, base: str) -> None:
        self.base = base

    def set_parameter(self, key: str, value: Any) -> None:
        self.parameters[key] = value

    def set_system(self, system: str) -> None:
        self.system = system

    def set_template(self, template: str) -> None:
        self.template = template

    def set_adapter(self, adapter: str) -> None:
        self.adapter = adapter

    def set_license(self, license_text: str) -> None:
        self.license = license_text

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def add_command(self, command: str) -> None:
        self.commands.append(command)

    def render(self) -> str:
        lines: list[str] = []
        if self.base:
            lines.append(f"FROM {self.base}")
        for k, v in self.parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        if self.system:
            lines.append(f'SYSTEM """{self.system}"""')
        if self.template:
            lines.append(f'TEMPLATE """{self.template}"""')
        if self.adapter:
            lines.append(f"ADAPTER {self.adapter}")
        if self.license:
            lines.append(f'LICENSE """{self.license}"""')
        for msg in self.messages:
            lines.append(f'MESSAGE {msg["role"]} """{msg["content"]}"""')
        lines.extend(self.commands)
        return "\n".join(lines)

    @classmethod
    def from_file(cls, filepath: str) -> "ModelFile":
        import re

        model = cls()
        multiline_fields = {"SYSTEM", "TEMPLATE", "LICENSE"}
        current_field = None
        current_value = []
        with Path(filepath).open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Multiline block start
                match = re.match(r'^(SYSTEM|TEMPLATE|LICENSE)\s+"""(.*)$', stripped)
                if match:
                    current_field = match.group(1)
                    rest = match.group(2)
                    if rest.endswith('"""'):
                        value = rest[:-3]
                        setattr(model, current_field.lower(), value)
                        current_field = None
                    else:
                        current_value = [rest]
                    continue
                # Multiline block end
                if current_field and stripped.endswith('"""'):
                    current_value.append(stripped[:-3])
                    value = "\n".join(current_value)
                    setattr(model, current_field.lower(), value)
                    current_field = None
                    current_value = []
                    continue
                if current_field:
                    current_value.append(stripped)
                    continue
                # Single line fields
                if stripped.startswith("FROM "):
                    model.base = stripped[5:].strip()
                elif stripped.startswith("PARAMETER "):
                    _, key, val = stripped.split(" ", 2)
                    model.parameters[key] = val
                elif stripped.startswith("ADAPTER "):
                    model.adapter = stripped[8:].strip()
                elif stripped.startswith("MESSAGE "):
                    msg_match = re.match(r'^MESSAGE\s+(\w+)\s+"""(.*)"""$', stripped)
                    if msg_match:
                        role, content = msg_match.groups()
                        model.add_message(role, content)
                    else:
                        # Fallback: try to parse without triple quotes
                        parts = stripped.split(" ", 2)
                        if len(parts) == 3:
                            model.add_message(parts[1], parts[2].strip('"'))
                else:
                    # Arbitrary command
                    model.commands.append(stripped)
        return model
