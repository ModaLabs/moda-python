"""Context managers for Moda SDK.

This module provides context managers for setting conversation and user
context that will be automatically applied to all LLM calls within the
context block.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from traceloop.sdk.conversation import (
    _set_conversation_id,
    _set_user_id,
    get_conversation_id,
    get_user_id,
)


@contextmanager
def set_conversation_id(conversation_id: str) -> Generator[None, None, None]:
    """Context manager for explicitly setting a conversation ID.

    Use this when you want to override the automatic conversation ID
    computation with an explicit ID.

    Example:
        with moda.set_conversation_id("my-custom-conversation-123"):
            client.chat.completions.create(...)
            # All calls here will use "my-custom-conversation-123"

    Args:
        conversation_id: The conversation ID to use within this context.

    Yields:
        None
    """
    previous = get_conversation_id()
    try:
        _set_conversation_id(conversation_id)
        yield
    finally:
        _set_conversation_id(previous)


@contextmanager
def set_user_id(user_id: str) -> Generator[None, None, None]:
    """Context manager for setting the user ID.

    Use this to associate LLM calls with a specific user for analytics
    and attribution.

    Example:
        with moda.set_user_id("user-456"):
            client.chat.completions.create(...)
            # All calls here will be attributed to "user-456"

    Args:
        user_id: The user ID to associate with calls in this context.

    Yields:
        None
    """
    previous = get_user_id()
    try:
        _set_user_id(user_id)
        yield
    finally:
        _set_user_id(previous)


def set_conversation_id_value(conversation_id: Optional[str]) -> None:
    """Set the conversation ID without using a context manager.

    This is useful when you want to set a conversation ID that persists
    across multiple operations without nesting.

    Args:
        conversation_id: The conversation ID to set, or None to clear.
    """
    _set_conversation_id(conversation_id)


def set_user_id_value(user_id: Optional[str]) -> None:
    """Set the user ID without using a context manager.

    This is useful when you want to set a user ID that persists
    across multiple operations without nesting.

    Args:
        user_id: The user ID to set, or None to clear.
    """
    _set_user_id(user_id)
