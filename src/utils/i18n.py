"""Localization and translation infrastructure using Python's gettext module."""

from __future__ import annotations

import contextvars
import gettext
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ContextVar containing the active gettext Translation object for the request.
# Defaults to NullTranslations (which returns English/raw strings).
_current_translation: contextvars.ContextVar[
    gettext.NullTranslations | gettext.GNUTranslations
] = contextvars.ContextVar("current_translation", default=gettext.NullTranslations())


def get_translation() -> gettext.NullTranslations | gettext.GNUTranslations:
    """Return the active translation object."""
    return _current_translation.get()


def _(message: str) -> str:
    """Translate message string using the active translation catalog.

    This function is bound globally or imported wherever localization is needed.
    """
    return get_translation().gettext(message)


def set_locale(lang: str) -> contextvars.Token:
    """Set the active language catalog for the current execution context.

    Returns a Token that can be used to restore the previous translation.
    """
    localedir = Path(__file__).parent.parent / "locale"
    try:
        translation = gettext.translation(
            domain="messages",
            localedir=str(localedir),
            languages=[lang],
            fallback=True,
        )
    except Exception:
        logger.debug("Failed to load translation catalog for %s, falling back", lang)
        translation = gettext.NullTranslations()

    return _current_translation.set(translation)
