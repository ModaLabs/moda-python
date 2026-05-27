import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from traceloop.sdk.prompts.client import PromptNotFoundError, PromptRegistryClient
from traceloop.sdk.tracing.tracing import set_managed_prompt_tracing_context


DEFAULT_PROMPT_GLOBS = ("prompts/**/*.prompt.md", "prompts/**/*.prompt.json",
                        "prompts/**/*.prompt.yml", "prompts/**/*.prompt.yaml")


def get_prompt(key, **args):
    return PromptRegistryClient().render_prompt(key, **args)


class PromptHandle:
    def __init__(self, key: str, **options):
        self.key = key
        self.options = options

    def render(self, variables: Optional[Dict[str, Any]] = None, **options):
        variables = variables or {}
        render_options = {**self.options, **options}

        try:
            return PromptRegistryClient().render_prompt(
                self.key,
                variables=variables,
                **render_options,
            )
        except PromptNotFoundError:
            pass

        definition = _load_local_prompt(self.key)
        content = _render_template(definition["content"], variables)
        system_prompt = definition.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": _render_template(system_prompt, variables)})
        messages.append({"role": "user", "content": content})

        set_managed_prompt_tracing_context(
            self.key,
            None,
            definition.get("name"),
            definition["content_hash"],
            variables,
            definition.get("prompt_id"),
            definition.get("version_id"),
        )

        params = {"messages": messages}
        if definition.get("model"):
            params["model"] = definition["model"]
        return params


def prompt(key: str, **options) -> PromptHandle:
    return PromptHandle(key, **options)


def _load_local_prompt(key: str) -> Dict[str, Any]:
    root = _find_project_root(Path.cwd())
    lock = _read_lock(root)
    globs = _read_prompt_paths(root) or DEFAULT_PROMPT_GLOBS

    prompt_files: List[Path] = []
    seen = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                prompt_files.append(path)

    for path in prompt_files:
        definition = _parse_prompt_file(root, path)
        if definition["key"] == key:
            locked = lock.get("prompts", {}).get(key, {})
            definition["prompt_id"] = locked.get("promptId")
            definition["version_id"] = locked.get("versionId")
            definition["content_hash"] = locked.get("contentHash") or definition["content_hash"]
            definition["source_path"] = locked.get("sourcePath") or definition["source_path"]
            return definition

    raise PromptNotFoundError(f"Prompt {key} does not exist")


def _find_project_root(cwd: Path) -> Path:
    current = cwd
    while True:
        if (current / ".moda" / "prompts.yml").exists() or (current / ".moda" / "prompts.lock.json").exists():
            return current
        if current.parent == current:
            return cwd
        current = current.parent


def _read_lock(root: Path) -> Dict[str, Any]:
    lock_path = root / ".moda" / "prompts.lock.json"
    if not lock_path.exists():
        return {}
    return json.loads(lock_path.read_text())


def _read_prompt_paths(root: Path) -> List[str]:
    config_path = root / ".moda" / "prompts.yml"
    if not config_path.exists():
        return []
    paths: List[str] = []
    in_list = False
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if in_list and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                paths.append(value)
            continue
        in_list = False
        if ":" in line:
            field, _, value = line.partition(":")
            if field.strip() == "prompt_paths":
                remainder = value.strip()
                if remainder.startswith("[") and remainder.endswith("]"):
                    inner = remainder[1:-1]
                    for item in inner.split(","):
                        cleaned = item.strip().strip("\"'")
                        if cleaned:
                            paths.append(cleaned)
                elif remainder:
                    paths.append(remainder.strip("\"'"))
                else:
                    in_list = True
    return paths


def _parse_prompt_file(root: Path, path: Path) -> Dict[str, Any]:
    raw = path.read_text()
    source_path = str(path.relative_to(root))

    if path.suffix == ".json":
        parsed = json.loads(raw)
        prompt_key = parsed.get("key") or _key_from_path(source_path)
        content = parsed.get("content") or json.dumps(parsed.get("messages", []), indent=2)
        return _normalize_definition({**parsed, "key": prompt_key, "source_path": source_path, "content": content})

    frontmatter, body = _parse_frontmatter(raw)
    prompt_key = frontmatter.get("key") or _key_from_path(source_path)
    return _normalize_definition({
        "key": prompt_key,
        "name": frontmatter.get("name") or prompt_key,
        "source_path": source_path,
        "content": body.strip(),
        "system_prompt": frontmatter.get("system_prompt") or frontmatter.get("systemPrompt"),
        "model": frontmatter.get("model"),
    })


def _normalize_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    stable = json.dumps({
        "key": definition["key"],
        "content": definition.get("content"),
        "system_prompt": definition.get("system_prompt"),
        "messages": definition.get("messages", []),
        "model": definition.get("model"),
    }, sort_keys=True)
    definition["name"] = definition.get("name") or definition["key"]
    definition["content_hash"] = definition.get("content_hash") or hashlib.sha256(stable.encode()).hexdigest()
    return definition


def _parse_frontmatter(raw: str):
    if not raw.startswith("---"):
        return {}, raw
    match = re.search(r"\n---\s*\n", raw[3:])
    if not match:
        return {}, raw
    end = match.start() + 3
    frontmatter_text = raw[3:end].strip()
    body = raw[end + len(match.group(0)):]
    frontmatter = {}
    for line in frontmatter_text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            frontmatter[parts[0].strip()] = parts[1].strip().strip("\"'")
    return frontmatter, body


def _key_from_path(source_path: str) -> str:
    key = re.sub(r"^prompts/", "", source_path)
    key = re.sub(r"\.prompt\.(md|json|ya?ml)$", "", key)
    return key.replace(os.sep, ".").replace("/", ".")


def _render_template(value: str, variables: Dict[str, Any]) -> str:
    def replace(match):
        current: Any = variables
        for part in match.group(1).split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return match.group(0)
        return str(current)

    return re.sub(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}", replace, value)
