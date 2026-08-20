from pathlib import Path
import urllib.request
import json

from PySide6.QtCore import QThread, Signal

_PROJECT_ROOT = Path(__file__).parent.parent
_REPO = "SebastianVanDijk/organoid_segmenter"
_API_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"


def _version_path():
    return _PROJECT_ROOT / "miscellaneous" / "version.txt"


def _read_local_version():
    # version.txt is a LOCAL runtime file (gitignored), not committed. It records
    # which release tag is currently installed. Returns None when absent (a fresh
    # install that has not been bootstrapped yet).
    p = _version_path()
    if not p.exists():
        return None
    return p.read_text().strip() or None


def _write_local_version(version):
    # Store the numeric version (no leading "v"), matching release tag comparison.
    p = _version_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(version).strip().lstrip("v") + "\n")


def _read_token():
    p = _PROJECT_ROOT / "github_token.txt"
    return p.read_text().strip() if p.exists() else None


def _parse_version(v):
    v = v.lstrip("v")
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


class UpdateChecker(QThread):
    update_available = Signal(str, str, str)   # (new_version, zipball_url, html_url)
    up_to_date = Signal(str)                    # (installed_version)
    check_failed = Signal(str)

    def run(self):
        try:
            req = urllib.request.Request(_API_URL, headers={"Accept": "application/vnd.github+json"})
            token = _read_token()
            if token:
                req.add_header("Authorization", f"token {token}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            tag = data.get("tag_name", "")
            zipball = data.get("zipball_url", "")
            html_url = data.get("html_url", "")
            if not tag:
                self.check_failed.emit("No tag_name in release response")
                return

            local = _read_local_version()
            if local is None:
                # Fresh install: adopt the current latest release as the baseline
                # so we don't prompt an "update" to the version already installed.
                _write_local_version(tag)
                self.up_to_date.emit(tag.lstrip("v"))
                return

            if _parse_version(tag) > _parse_version(local):
                self.update_available.emit(tag.lstrip("v"), zipball, html_url)
            else:
                self.up_to_date.emit(local)
        except Exception as e:
            self.check_failed.emit(str(e))
