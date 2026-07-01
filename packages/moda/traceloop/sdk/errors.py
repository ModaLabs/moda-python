"""Loud-fail contract for the Moda Python SDK.

This module mirrors the Node SDK's loud-fail contract (workstream WS-SDK-NODE)
field-for-field so both SDKs agree on how the SDK reacts to misconfiguration.

The on-error mode has three settled values (identical to Node):

    - ``"silent"``: do nothing (optionally debug-log). The SDK stays quiet.
    - ``"warn"``:   emit a warning and continue (**default**). Non-disruptive.
    - ``"throw"``:  raise :class:`ModaConfigError`.

COEXISTENCE is a first principle: the default is ``"warn"`` so the SDK never
crashes a caller's existing app on misconfiguration unless the caller
explicitly opts into ``"throw"``. The ``"throw"`` path exists so the onboarding
VERIFY stage can be honest ("never silently partial").

This is the SERIAL ROOT of WS-SDK-PY: sibling issues import ``OnError``,
``ModaConfigError`` and ``handle_config_issue`` from here.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional, Type, Union

from colorama import Fore

# Environment variable that overrides the default loud-fail mode.
ON_ERROR_ENV_VAR = "MODA_ON_ERROR"


class OnError(str, Enum):
    """How the SDK reacts to a configuration problem.

    A ``str`` enum so ``OnError.WARN == "warn"`` and callers may pass either the
    enum member or a raw string. Values mirror the Node contract exactly.
    """

    SILENT = "silent"
    WARN = "warn"
    THROW = "throw"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# The default mode when nothing is configured. Mirrors Node's default.
DEFAULT_ON_ERROR = OnError.WARN


class ModaConfigError(ValueError):
    """Raised when the SDK is misconfigured and ``on_error='throw'``.

    Mirrors the Node contract's config error type. Sub-types below narrow the
    specific failure so callers/tests can catch them precisely while still being
    able to catch the base ``ModaConfigError``.

    Subclasses :class:`ValueError` so that historical callers/tests that catch a
    bare ``ValueError`` for a misconfiguration (e.g. the credential-bridge's
    "API key is required" path) keep working after the missing-key decision was
    unified onto the loud-fail contract — ``pytest.raises(ValueError)`` still
    catches a ``ModaMissingApiKeyError`` raised under ``on_error='throw'``.
    """


class ModaMissingApiKeyError(ModaConfigError):
    """Raised when no Moda API key is available and ``on_error='throw'``."""


class ModaExporterError(ModaConfigError):
    """Raised when the SDK cannot attach/flush its exporter and ``on_error='throw'``.

    Covers an un-attachable/``None`` tracer provider and hard export failures so
    the onboarding VERIFY stage never sees a silently-broken tracing pipeline.
    """


def coerce_on_error(value: Union["OnError", str, None]) -> OnError:
    """Normalize a raw value (enum / string / ``None``) into an :class:`OnError`.

    Unknown strings and ``None`` fall back to :data:`DEFAULT_ON_ERROR` rather
    than raising — resolving the mode must never itself become a loud failure.
    """
    if value is None:
        return DEFAULT_ON_ERROR
    if isinstance(value, OnError):
        return value
    try:
        return OnError(str(value).strip().lower())
    except ValueError:
        return DEFAULT_ON_ERROR


def resolve_on_error(explicit: Union["OnError", str, None] = None) -> OnError:
    """Resolve the effective loud-fail mode.

    Precedence: **explicit arg > ``MODA_ON_ERROR`` env var > default ``'warn'``**.
    """
    if explicit is not None:
        return coerce_on_error(explicit)

    env_value = os.getenv(ON_ERROR_ENV_VAR)
    if env_value:
        return coerce_on_error(env_value)

    return DEFAULT_ON_ERROR


def handle_config_issue(
    message: str,
    *,
    on_error: Union["OnError", str] = DEFAULT_ON_ERROR,
    logger: Optional[logging.Logger] = None,
    color: str = Fore.RED,
    error_cls: Type[ModaConfigError] = ModaConfigError,
) -> None:
    """Dispatch a configuration problem according to the loud-fail mode.

    - ``silent``: optionally debug-log (if ``logger`` given); otherwise nothing.
    - ``warn``:   emit ``logging.warning`` (if ``logger`` given) plus a colorized
      stderr/stdout line, reusing the existing ``colorama`` pattern. Returns
      without raising — the SDK keeps running.
    - ``throw``:  raise ``error_cls(message)`` (defaults to
      :class:`ModaConfigError`).

    Args:
        message: Human-readable description of the misconfiguration.
        on_error: The resolved loud-fail mode (enum or raw string).
        logger: Optional logger; used for ``debug`` (silent) / ``warning`` (warn).
        color: ``colorama`` colour for the printed warn line (defaults to red,
            matching the existing missing-key message).
        error_cls: Exception class raised in ``throw`` mode.
    """
    mode = coerce_on_error(on_error)

    if mode is OnError.THROW:
        raise error_cls(message)

    if mode is OnError.SILENT:
        if logger is not None:
            logger.debug(message)
        return

    # WARN (default): non-disruptive — log + print, then return.
    if logger is not None:
        logger.warning(message)
    print(color + message + Fore.RESET)


__all__ = [
    "OnError",
    "DEFAULT_ON_ERROR",
    "ON_ERROR_ENV_VAR",
    "ModaConfigError",
    "ModaMissingApiKeyError",
    "ModaExporterError",
    "coerce_on_error",
    "resolve_on_error",
    "handle_config_issue",
]
