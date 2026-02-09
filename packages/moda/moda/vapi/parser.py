"""Parsing functions for extracting data from Vapi end-of-call reports."""

from typing import Optional

from moda.vapi.types import (
    VapiCallArtifact,
    VapiMessage,
    VapiCost,
    VapiTransfer,
    VapiTurnLatency,
    ExtractedTurn,
    ExtractedToolCall,
)


def extract_turns(artifact: Optional[VapiCallArtifact]) -> list[ExtractedTurn]:
    """Extract assistant turns from the message history.

    A turn consists of a user message followed by an assistant response.

    Args:
        artifact: The call artifact containing messages and metrics.

    Returns:
        List of extracted turns with user/assistant messages and timing data.
    """
    if artifact is None:
        return []

    messages = artifact.get("messages")
    if not messages:
        return []

    turns: list[ExtractedTurn] = []
    turn_latencies = (
        artifact.get("performanceMetrics", {}).get("turnLatencies") or []
    )
    turn_index = 0

    for i, message in enumerate(messages):
        # Look for assistant messages (these represent LLM turns)
        if message.get("role") == "assistant":
            # Find the preceding user message if any
            user_message: Optional[VapiMessage] = None
            for j in range(i - 1, -1, -1):
                if messages[j].get("role") == "user":
                    user_message = messages[j]
                    break
                # Stop if we hit another assistant message
                if messages[j].get("role") == "assistant":
                    break

            # Find timing data for this turn
            timing: Optional[VapiTurnLatency] = None
            # Try to find by turnIndex field first
            for t in turn_latencies:
                if t.get("turnIndex") == turn_index:
                    timing = t
                    break
            # Fall back to array position
            if timing is None and turn_index < len(turn_latencies):
                timing = turn_latencies[turn_index]

            turn: ExtractedTurn = {
                "index": turn_index,
                "assistantMessage": message,
            }
            if user_message is not None:
                turn["userMessage"] = user_message
            if timing is not None:
                turn["timing"] = timing

            turns.append(turn)
            turn_index += 1

    return turns


def extract_tool_calls(artifact: Optional[VapiCallArtifact]) -> list[ExtractedToolCall]:
    """Extract tool calls from the message history.

    Looks for tool_calls in assistant messages and corresponding tool result messages.

    Args:
        artifact: The call artifact containing messages.

    Returns:
        List of extracted tool calls with parameters and results.
    """
    if artifact is None:
        return []

    messages = artifact.get("messages")
    if not messages:
        return []

    tool_calls: list[ExtractedToolCall] = []
    tool_index = 0

    # Build a map of tool call ID to result
    tool_results: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        if role in ("tool", "function") and message.get("tool_call_id"):
            tool_results[message["tool_call_id"]] = message.get("content", "")

    # Extract tool calls from assistant messages
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                extracted: ExtractedToolCall = {
                    "name": tool_call["function"]["name"],
                    "index": tool_index,
                }
                if tool_call.get("id"):
                    extracted["id"] = tool_call["id"]
                    if tool_call["id"] in tool_results:
                        extracted["result"] = tool_results[tool_call["id"]]
                if tool_call["function"].get("arguments"):
                    extracted["parameters"] = tool_call["function"]["arguments"]

                tool_calls.append(extracted)
                tool_index += 1

    # Also look for standalone tool role messages that might not have corresponding tool_calls
    for message in messages:
        if (
            message.get("role") == "tool"
            and message.get("name")
            and not message.get("tool_call_id")
        ):
            # This is a tool result without a corresponding tool_call - still track it
            extracted = ExtractedToolCall(
                name=message["name"],
                index=tool_index,
            )
            if message.get("content"):
                extracted["result"] = message["content"]

            tool_calls.append(extracted)
            tool_index += 1

    return tool_calls


def extract_transfers(artifact: Optional[VapiCallArtifact]) -> list[VapiTransfer]:
    """Extract squad transfer records from the artifact.

    Args:
        artifact: The call artifact containing transfer records.

    Returns:
        List of transfer records.
    """
    if artifact is None:
        return []

    transfers = artifact.get("transfers")
    if not transfers:
        return []

    return transfers


def extract_costs(costs: Optional[list[VapiCost]]) -> dict[str, float]:
    """Extract and normalize cost breakdown from the call.

    Args:
        costs: List of cost entries from the call.

    Returns:
        Dictionary mapping cost type to amount.
    """
    cost_map: dict[str, float] = {}

    if not costs:
        return cost_map

    for cost in costs:
        # Normalize cost type names
        normalized_type = _normalize_cost_type(cost["type"])

        # Accumulate costs of the same type
        existing = cost_map.get(normalized_type, 0)
        cost_map[normalized_type] = existing + cost["cost"]

    return cost_map


def _normalize_cost_type(cost_type: str) -> str:
    """Normalize cost type names to consistent keys.

    Args:
        cost_type: The original cost type string.

    Returns:
        Normalized cost type key.
    """
    lower = cost_type.lower()

    # Map various names to standard types
    if any(term in lower for term in ("model", "llm", "chat")):
        return "model"
    if any(term in lower for term in ("transcri", "stt", "speech-to-text")):
        return "transcriber"
    if any(term in lower for term in ("voice", "tts", "text-to-speech")):
        return "voice"

    return lower


def get_turn_timing(
    artifact: Optional[VapiCallArtifact], turn_index: int
) -> Optional[VapiTurnLatency]:
    """Get timing information for a specific turn.

    Args:
        artifact: The call artifact containing performance metrics.
        turn_index: The index of the turn to get timing for.

    Returns:
        Timing data for the turn, or None if not available.
    """
    if artifact is None:
        return None

    performance_metrics = artifact.get("performanceMetrics")
    if not performance_metrics:
        return None

    latencies = performance_metrics.get("turnLatencies")
    if not latencies:
        return None

    # Try to find by turnIndex field first
    for t in latencies:
        if t.get("turnIndex") == turn_index:
            return t

    # Fall back to array position
    if turn_index < len(latencies):
        return latencies[turn_index]

    return None
