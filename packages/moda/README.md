# moda-ai

Moda's Python SDK allows you to easily start monitoring and debugging your LLM execution with automatic conversation threading. Tracing is done in a non-intrusive way, built on top of OpenTelemetry.

## Installation

```bash
pip install moda-ai
```

## Quick Start

```python
import moda
from openai import OpenAI

moda.init("moda_xxx")

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke about observability"}],
)

print(response.choices[0].message.content)
```
