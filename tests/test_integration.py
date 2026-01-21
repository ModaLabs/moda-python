"""Integration tests for Moda SDK with mock LLM providers."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Add the packages to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda-openai'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'moda-anthropic'))

from traceloop.sdk.conversation import compute_conversation_id, get_user_id
from traceloop.sdk.context import set_conversation_id, set_user_id


class TestOpenAIConversationTracking:
    """Tests for OpenAI instrumentation conversation tracking."""

    def test_compute_conversation_id_openai_format(self):
        """Test conversation ID computation with OpenAI message format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        conv_id = compute_conversation_id(messages)

        assert conv_id.startswith("conv_")
        assert len(conv_id) == 21  # "conv_" + 16 chars

    def test_same_conversation_multiple_turns(self):
        """Test that multi-turn conversations maintain same ID."""
        # Turn 1
        messages_turn1 = [
            {"role": "user", "content": "Hello!"},
        ]

        # Turn 2 - same conversation
        messages_turn2 = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What's the weather like?"},
        ]

        id1 = compute_conversation_id(messages_turn1)
        id2 = compute_conversation_id(messages_turn2)

        assert id1 == id2, "Same first message should produce same conversation ID"

    def test_context_manager_override_works(self):
        """Test that context manager override takes precedence."""
        messages = [{"role": "user", "content": "Hello"}]
        custom_id = "my-custom-conversation-id"

        with set_conversation_id(custom_id):
            result = compute_conversation_id(messages)
            assert result == custom_id


class TestAnthropicConversationTracking:
    """Tests for Anthropic instrumentation conversation tracking."""

    def test_anthropic_format_with_separate_system(self):
        """Test conversation ID with Anthropic's separate system parameter."""
        # Anthropic uses system as a separate kwarg, not in messages
        # The instrumentor normalizes this before computing ID

        # Simulate what the instrumentor does
        kwargs = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }

        # Normalize like the instrumentor does
        normalized = []
        if kwargs.get("system"):
            normalized.append({"role": "system", "content": kwargs["system"]})
        normalized.extend(kwargs.get("messages", []))

        conv_id = compute_conversation_id(normalized)
        assert conv_id.startswith("conv_")

    def test_anthropic_system_as_list(self):
        """Test Anthropic system prompt as list of content blocks."""
        kwargs = {
            "system": [
                {"type": "text", "text": "You are a coding assistant."},
            ],
            "messages": [{"role": "user", "content": "Write hello world"}],
        }

        # Normalize like the instrumentor does
        normalized = []
        system_content = kwargs["system"]
        if isinstance(system_content, list):
            text_parts = []
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            normalized.append({"role": "system", "content": "".join(text_parts)})
        normalized.extend(kwargs.get("messages", []))

        conv_id = compute_conversation_id(normalized)
        assert conv_id.startswith("conv_")

    def test_anthropic_no_system(self):
        """Test Anthropic messages without system prompt."""
        kwargs = {"messages": [{"role": "user", "content": "Hello!"}]}

        normalized = kwargs.get("messages", [])
        conv_id = compute_conversation_id(normalized)

        assert conv_id.startswith("conv_")


class TestUserIdTracking:
    """Tests for user ID tracking."""

    def test_user_id_in_context(self):
        """Test user ID retrieval from context."""
        assert get_user_id() is None

        with set_user_id("user-123"):
            assert get_user_id() == "user-123"

        assert get_user_id() is None

    def test_nested_user_ids(self):
        """Test nested user ID contexts."""
        with set_user_id("user-outer"):
            assert get_user_id() == "user-outer"

            with set_user_id("user-inner"):
                assert get_user_id() == "user-inner"

            assert get_user_id() == "user-outer"


class TestSpanAttributeIntegration:
    """Tests for span attribute setting integration."""

    def test_moda_attributes_set_correctly(self):
        """Test that moda.conversation_id is computed correctly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        conv_id = compute_conversation_id(messages)

        # Verify the format
        assert conv_id.startswith("conv_")
        assert len(conv_id) == 21

        # Verify it's deterministic
        conv_id_2 = compute_conversation_id(messages)
        assert conv_id == conv_id_2

    def test_conversation_id_hash_is_stable(self):
        """Test that conversation ID hash is stable across runs."""
        messages = [{"role": "user", "content": "Test message"}]

        # Run multiple times
        ids = [compute_conversation_id(messages) for _ in range(10)]

        # All should be identical
        assert len(set(ids)) == 1

    def test_different_conversations_different_ids(self):
        """Test that different conversations get different IDs."""
        conv1_messages = [{"role": "user", "content": "Conversation 1 start"}]
        conv2_messages = [{"role": "user", "content": "Conversation 2 start"}]

        id1 = compute_conversation_id(conv1_messages)
        id2 = compute_conversation_id(conv2_messages)

        assert id1 != id2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_string(self):
        """Test message with empty content string."""
        messages = [{"role": "user", "content": ""}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")

    def test_whitespace_only_content(self):
        """Test message with whitespace-only content."""
        messages = [{"role": "user", "content": "   "}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")

    def test_unicode_content(self):
        """Test message with unicode content."""
        messages = [{"role": "user", "content": "Hello! \u4f60\u597d\uff01 \ud83d\udc4b"}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")

    def test_very_long_content(self):
        """Test message with very long content."""
        long_content = "x" * 100000
        messages = [{"role": "user", "content": long_content}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")
        assert len(conv_id) == 21  # Fixed length regardless of input

    def test_special_characters_in_content(self):
        """Test message with special characters."""
        messages = [{"role": "user", "content": "Hello <script>alert('xss')</script>"}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")

    def test_json_in_content(self):
        """Test message with JSON content."""
        messages = [{"role": "user", "content": '{"key": "value", "nested": {"a": 1}}'}]
        conv_id = compute_conversation_id(messages)
        assert conv_id.startswith("conv_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
