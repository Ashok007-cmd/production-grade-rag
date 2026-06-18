#!/usr/bin/env python3
"""Find and compile gettext translation catalogs (.po -> .mo).

Runs `msgfmt` on all .po files under the src/locale directory.
"""

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compile_locales() -> None:
    locale_dir = Path(__file__).parent.parent / "src" / "locale"
    if not locale_dir.exists():
        logger.error("Locale directory not found at: %s", locale_dir)
        return

    po_files = list(locale_dir.glob("**/messages.po"))
    if not po_files:
        logger.warning("No messages.po files found in %s", locale_dir)
        return

    for po_path in po_files:
        mo_path = po_path.with_suffix(".mo")
        logger.info(
            "Compiling %s -> %s",
            po_path.relative_to(locale_dir.parent),
            mo_path.relative_to(locale_dir.parent),
        )
        try:
            subprocess.run(
                ["msgfmt", "-o", str(mo_path), str(po_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Successfully compiled: %s", mo_path.name)
        except subprocess.CalledProcessError as err:
            logger.error("Failed to compile %s: %s", po_path, err.stderr)
            raise err


if __name__ == "__main__":
    compile_locales()
