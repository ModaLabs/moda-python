"""Conversation tracking for Moda SDK.

This module provides automatic conversation threading by computing stable
conversation IDs based on message content.
"""

import hashlib
import json
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variables for conversation and user tracking
_conversation_id_var: ContextVar[Optional[str]] = ContextVar(
    "conversation_id", default=None
)
_user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def compute_conversation_id(messages: list[dict]) -> str:
    """Compute a stable conversation ID from messages.

    The conversation ID is computed by hashing the first user message
    combined with the system prompt (if present). This ensures that
    conversations with the same starting context get the same ID.

    If an explicit conversation ID has been set via set_conversation_id(),
    that ID is returned instead.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        A conversation ID string in the format 'conv_<hash>' or 'conv_<uuid>'.
    """
    # Check for explicit override first
    ctx_id = _conversation_id_var.get()
    if ctx_id:
        return ctx_id

    # Find the first user message
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if not first_user:
        return f"conv_{uuid.uuid4().hex[:16]}"

    # Find system message if present
    system = next((m for m in messages if m.get("role") == "system"), None)

    # Build seed for hashing
    seed = json.dumps(
        {
            "system": _extract_content(system) if system else None,
            "first_user": _extract_content(first_user),
        },
        sort_keys=True,
    )

    return f"conv_{hashlib.sha256(seed.encode()).hexdigest()[:16]}"


def _extract_content(message: dict) -> Optional[str]:
    """Extract text content from a message.

    Handles both simple string content and complex content arrays.
    """
    if not message:
        return None

    content = message.get("content")
    if content is None:
        return None

    if isinstance(content, str):
        return content

    # Handle content arrays (e.g., for multimodal messages)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif "text" in part:
                    text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts) if text_parts else None

    return str(content)


def get_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context.

    Returns:
        The conversation ID if set, None otherwise.
    """
    return _conversation_id_var.get()


def get_user_id() -> Optional[str]:
    """Get the current user ID from context.

    Returns:
        The user ID if set, None otherwise.
    """
    return _user_id_var.get()


def _set_conversation_id(conversation_id: Optional[str]) -> None:
    """Internal function to set conversation ID in context."""
    _conversation_id_var.set(conversation_id)


def _set_user_id(user_id: Optional[str]) -> None:
    """Internal function to set user ID in context."""
    _user_id_var.set(user_id)
