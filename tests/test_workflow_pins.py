"""Supply-chain guards for the GitHub Actions workflows."""

import re
from pathlib import Path

import pytest

GITHUB_DIR = Path(__file__).resolve().parents[1] / ".github"
WORKFLOWS = sorted((GITHUB_DIR / "workflows").glob("*.yml"))

USES = re.compile(r"^\s*(?:- )?uses:\s*(.+?)\s*$", re.MULTILINE)
# Commit SHA plus the tag it resolves to, which Dependabot keeps in sync.
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40} # v?\d+\.\d+\.\d+$")
SETUP_UV = re.compile(r"uses: astral-sh/setup-uv@.*\n\s+with:\n\s+enable-cache: false\n")


def workflow_id(path):
    return path.name


def jobs(path):
    """Map each job name in a workflow to the text of its definition."""
    _, _, body = path.read_text().partition("\njobs:\n")
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def test_workflows_found():
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=workflow_id)
def test_actions_pinned_to_commit_sha(path):
    refs = USES.findall(path.read_text())
    unpinned = [ref for ref in refs if not ref.startswith("./") and not PINNED.match(ref)]
    assert not unpinned


@pytest.mark.parametrize("path", WORKFLOWS, ids=workflow_id)
def test_no_remote_script_piped_to_shell(path):
    assert not re.search(r"(curl|wget)\b.*\|\s*(ba|z)?sh\b", path.read_text())


@pytest.mark.parametrize("path", WORKFLOWS, ids=workflow_id)
def test_setup_uv_cache_disabled(path):
    # The default cache would let a release job restore a cache written by an
    # earlier run. The curl installer never cached, so keep it that way.
    text = path.read_text()
    assert text.count("astral-sh/setup-uv@") == len(SETUP_UV.findall(text))


@pytest.mark.parametrize("path", WORKFLOWS, ids=workflow_id)
def test_id_token_only_on_publish_jobs(path):
    for name, job in jobs(path).items():
        publishes = "pypa/gh-action-pypi-publish@" in job
        assert ("id-token: write" in job.replace('"', "")) == publishes, name


def test_dependabot_updates_github_actions():
    config = (GITHUB_DIR / "dependabot.yml").read_text()
    assert 'package-ecosystem: "github-actions"' in config
