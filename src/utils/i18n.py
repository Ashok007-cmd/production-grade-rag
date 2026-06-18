"""Localization and translation infrastructure using Python's gettext module."""

from __future__ import annotations

import contextvars
import gettext
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ContextVar holding the active translation for the current context.
# None means "no translation set yet" — get_translation() falls back to NullTranslations.
_current_translation: contextvars.ContextVar[
    gettext.NullTranslations | gettext.GNUTranslations | None
] = contextvars.ContextVar("current_translation", default=None)

_NULL_TRANSLATIONS = gettext.NullTranslations()


def get_translation() -> gettext.NullTranslations | gettext.GNUTranslations:
    """Return the active translation object, falling back to NullTranslations."""
    return _current_translation.get() or _NULL_TRANSLATIONS


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
