"""Unit tests for conversation threading module."""

import pytest
import sys
import os

# Add the packages to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))

from traceloop.sdk.conversation import (
    compute_conversation_id,
    get_conversation_id,
    get_user_id,
    _set_conversation_id,
    _set_user_id,
    _extract_content,
)
from traceloop.sdk.context import (
    set_conversation_id,
    set_user_id,
    set_conversation_id_value,
    set_user_id_value,
)


class TestComputeConversationId:
    """Tests for compute_conversation_id function."""

    def test_same_first_message_same_id(self):
        """Same first user message should produce same conversation ID."""
        messages1 = [{"role": "user", "content": "Hello, how are you?"}]
        messages2 = [{"role": "user", "content": "Hello, how are you?"}]

        id1 = compute_conversation_id(messages1)
        id2 = compute_conversation_id(messages2)

        assert id1 == id2
        assert id1.startswith("conv_")

    def test_different_first_message_different_id(self):
        """Different first user messages should produce different IDs."""
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Goodbye"}]

        id1 = compute_conversation_id(messages1)
        id2 = compute_conversation_id(messages2)

        assert id1 != id2

    def test_system_prompt_changes_id(self):
        """System prompt should affect the conversation ID."""
        messages_no_system = [{"role": "user", "content": "Hello"}]
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        id1 = compute_conversation_id(messages_no_system)
        id2 = compute_conversation_id(messages_with_system)

        assert id1 != id2

    def test_different_system_prompts_different_ids(self):
        """Different system prompts should produce different IDs."""
        messages1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        messages2 = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Hello"},
        ]

        id1 = compute_conversation_id(messages1)
        id2 = compute_conversation_id(messages2)

        assert id1 != id2

    def test_multi_turn_same_first_message(self):
        """Multi-turn conversations with same first message should have same ID."""
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        id1 = compute_conversation_id(messages1)
        id2 = compute_conversation_id(messages2)

        assert id1 == id2

    def test_no_user_message_returns_random_id(self):
        """No user message should return a random UUID-based ID."""
        messages = [{"role": "assistant", "content": "Hello"}]

        id1 = compute_conversation_id(messages)
        id2 = compute_conversation_id(messages)

        assert id1.startswith("conv_")
        assert id2.startswith("conv_")
        # Random IDs should be different each time
        assert id1 != id2

    def test_empty_messages_returns_random_id(self):
        """Empty messages list should return a random ID."""
        id1 = compute_conversation_id([])
        id2 = compute_conversation_id([])

        assert id1.startswith("conv_")
        assert id2.startswith("conv_")
        assert id1 != id2

    def test_explicit_override_takes_precedence(self):
        """Explicit conversation ID override should take precedence."""
        messages = [{"role": "user", "content": "Hello"}]
        override_id = "custom-conv-123"

        _set_conversation_id(override_id)
        try:
            result = compute_conversation_id(messages)
            assert result == override_id
        finally:
            _set_conversation_id(None)

    def test_multimodal_content_array(self):
        """Content arrays (multimodal messages) should be handled."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]

        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")


class TestContextManagers:
    """Tests for context manager functions."""

    def test_set_conversation_id_context_manager(self):
        """set_conversation_id context manager should work correctly."""
        assert get_conversation_id() is None

        with set_conversation_id("test-conv-123"):
            assert get_conversation_id() == "test-conv-123"

        assert get_conversation_id() is None

    def test_set_user_id_context_manager(self):
        """set_user_id context manager should work correctly."""
        assert get_user_id() is None

        with set_user_id("user-456"):
            assert get_user_id() == "user-456"

        assert get_user_id() is None

    def test_nested_context_managers(self):
        """Nested context managers should restore previous values."""
        with set_conversation_id("outer"):
            assert get_conversation_id() == "outer"
            with set_conversation_id("inner"):
                assert get_conversation_id() == "inner"
            assert get_conversation_id() == "outer"

    def test_set_value_functions(self):
        """set_*_value functions should work without context manager."""
        set_conversation_id_value("my-conv")
        assert get_conversation_id() == "my-conv"

        set_user_id_value("my-user")
        assert get_user_id() == "my-user"

        # Clean up
        set_conversation_id_value(None)
        set_user_id_value(None)

        assert get_conversation_id() is None
        assert get_user_id() is None


class TestExtractContent:
    """Tests for _extract_content helper function."""

    def test_string_content(self):
        """String content should be returned as-is."""
        msg = {"role": "user", "content": "Hello world"}
        assert _extract_content(msg) == "Hello world"

    def test_none_content(self):
        """None content should return None."""
        msg = {"role": "user", "content": None}
        assert _extract_content(msg) is None

    def test_missing_content(self):
        """Missing content key should return None."""
        msg = {"role": "user"}
        assert _extract_content(msg) is None

    def test_list_content_with_text_blocks(self):
        """List content with text blocks should be concatenated."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        }
        assert _extract_content(msg) == "Hello world"

    def test_list_content_with_mixed_blocks(self):
        """List content with mixed blocks should extract text only."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this?"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ],
        }
        assert _extract_content(msg) == "What's in this?"

    def test_empty_message(self):
        """Empty message dict should return None."""
        assert _extract_content({}) is None
        assert _extract_content(None) is None


class TestPropertyStyleAPI:
    """Tests for property-style conversation/user ID access."""

    def setup_method(self):
        """Reset state before each test."""
        set_conversation_id_value(None)
        set_user_id_value(None)

    def teardown_method(self):
        """Clean up after each test."""
        set_conversation_id_value(None)
        set_user_id_value(None)

    def test_conversation_id_property_get_set(self):
        """Test property-style conversation ID get/set."""
        # Add moda module to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))
        import moda

        # Initially None
        assert moda.conversation_id is None

        # Set via property
        moda.conversation_id = "test_conv_123"
        assert moda.conversation_id == "test_conv_123"

        # Also accessible via getter function
        assert get_conversation_id() == "test_conv_123"

        # Clear via None
        moda.conversation_id = None
        assert moda.conversation_id is None

    def test_user_id_property_get_set(self):
        """Test property-style user ID get/set."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))
        import moda

        # Initially None
        assert moda.user_id is None

        # Set via property
        moda.user_id = "user_456"
        assert moda.user_id == "user_456"

        # Also accessible via getter function
        assert get_user_id() == "user_456"

        # Clear via None
        moda.user_id = None
        assert moda.user_id is None

    def test_property_and_context_manager_consistency(self):
        """Test that property and context manager APIs are consistent."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))
        import moda

        # Set via property
        moda.conversation_id = "prop_conv"
        assert moda.conversation_id == "prop_conv"

        # Getter function should see it
        assert get_conversation_id() == "prop_conv"

        # Context manager overrides temporarily
        with set_conversation_id("scoped_conv"):
            assert moda.conversation_id == "scoped_conv"

        # Back to property value
        assert moda.conversation_id == "prop_conv"

    def test_property_and_value_setter_consistency(self):
        """Test that property and set_*_value functions are consistent."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))
        import moda

        # Set via value function
        set_conversation_id_value("func_conv")
        assert moda.conversation_id == "func_conv"

        # Set via property
        moda.conversation_id = "prop_conv"
        assert get_conversation_id() == "prop_conv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
