#!/usr/bin/env python3
"""Script to automate package version bumping and CHANGELOG updating.

Usage:
    python scripts/bump_version.py --part [major|minor|patch] --message "Added some features"
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump package version and update CHANGELOG.md")
    parser.add_argument(
        "--part",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Part of the version to bump (default: patch)",
    )
    parser.add_argument(
        "--message",
        "-m",
        required=True,
        help="Release message / changelog description",
    )
    args = parser.parse_args()

    setup_file = Path("setup.py")
    changelog_file = Path("CHANGELOG.md")

    if not setup_file.exists():
        print("Error: setup.py not found in current directory.")
        return

    # Read current version from setup.py
    setup_content = setup_file.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*(["\'])([^"\']+)(["\'])', setup_content)
    if not match:
        print("Error: Could not find version string in setup.py.")
        return

    quote_char = match.group(1)
    current_version = match.group(2)

    try:
        major, minor, patch = map(int, current_version.split("."))
    except ValueError:
        print(f"Error: Version string '{current_version}' is not in semver format (X.Y.Z).")
        return

    # Bump version
    if args.part == "major":
        major += 1
        minor = 0
        patch = 0
    elif args.part == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    new_version = f"{major}.{minor}.{patch}"
    print(f"Bumping version from {current_version} to {new_version}...")

    # Write new version back to setup.py while preserving original quote style
    new_setup_content = re.sub(
        r'(version\s*=\s*)(["\'])([^"\']*)(["\'])',
        f"\\g<1>{quote_char}{new_version}{quote_char}",
        setup_content,
    )
    setup_file.write_text(new_setup_content, encoding="utf-8")
    print("Updated setup.py.")

    # Update CHANGELOG.md
    today = datetime.today().strftime("%Y-%m-%d")
    changelog_entry = f"## [{new_version}] - {today}\n\n- {args.message}\n\n"

    if changelog_file.exists():
        existing_changelog = changelog_file.read_text(encoding="utf-8")
        if "# Changelog" in existing_changelog:
            # Insert entry right after title header
            parts = existing_changelog.split("# Changelog\n\n", 1)
            if len(parts) == 2:
                new_changelog = f"# Changelog\n\n{changelog_entry}{parts[1]}"
            else:
                new_changelog = f"# Changelog\n\n{changelog_entry}{existing_changelog}"
        else:
            new_changelog = changelog_entry + existing_changelog
    else:
        new_changelog = (
            f"# Changelog\n\n"
            f"All notable changes to this project will be documented in this file.\n\n"
            f"{changelog_entry}"
        )

    changelog_file.write_text(new_changelog, encoding="utf-8")
    print("Updated CHANGELOG.md.")


if __name__ == "__main__":
    main()
