from pathlib import Path
import urllib.request
import json

from PySide2.QtCore import QThread, Signal

_PROJECT_ROOT = Path(__file__).parent.parent
_REPO = "SebastianVanDijk/organoid_segmenter"
_API_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"


def _read_local_version():
    p = _PROJECT_ROOT / "miscellaneous" / "version.txt"
    return p.read_text().strip() if p.exists() else "0.0.0"


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
    update_available = Signal(str, str)   # (new_version, zipball_url)
    up_to_date = Signal()
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
            if not tag:
                self.check_failed.emit("No tag_name in release response")
                return

            remote = _parse_version(tag)
            local = _parse_version(_read_local_version())
            if remote > local:
                self.update_available.emit(tag.lstrip("v"), zipball)
            else:
                self.up_to_date.emit()
        except Exception as e:
            self.check_failed.emit(str(e))
