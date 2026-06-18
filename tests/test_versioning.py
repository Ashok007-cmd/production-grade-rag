from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_bump_version_script(tmp_path: Path) -> None:
    # Setup mock setup.py
    setup_file = tmp_path / "setup.py"
    setup_file.write_text('setup(name="test", version="1.2.3")', encoding="utf-8")

    changelog_file = tmp_path / "CHANGELOG.md"

    # Find local scripts/bump_version.py path
    script_path = Path(__file__).parent.parent / "scripts" / "bump_version.py"

    # Run patch bump
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--part",
            "patch",
            "-m",
            "Patch release test",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Bumping version from 1.2.3 to 1.2.4" in result.stdout

    # Verify setup.py updated
    setup_content = setup_file.read_text(encoding="utf-8")
    assert 'version="1.2.4"' in setup_content

    # Verify CHANGELOG.md created and updated
    changelog_content = changelog_file.read_text(encoding="utf-8")
    assert "## [1.2.4]" in changelog_content
    assert "- Patch release test" in changelog_content


def test_bump_version_script_minor(tmp_path: Path) -> None:
    # Setup mock setup.py
    setup_file = tmp_path / "setup.py"
    setup_file.write_text("setup(name='test', version='2.5.9')", encoding="utf-8")

    script_path = Path(__file__).parent.parent / "scripts" / "bump_version.py"

    # Run minor bump
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--part",
            "minor",
            "-m",
            "Minor release test",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )

    # Verify setup.py updated (preserving single quotes)
    setup_content = setup_file.read_text(encoding="utf-8")
    assert "version='2.6.0'" in setup_content
