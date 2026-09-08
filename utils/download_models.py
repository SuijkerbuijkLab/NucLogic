"""Fetch the model weights, which ship as release assets rather than in the repo.

Assets are attached to their own tag so model downloads are independent of app
releases, and they are not part of the git history, so a clone or a zip download
arrives without them.
"""
from pathlib import Path
import urllib.request
import json

from PySide6.QtCore import QThread, Signal

from utils.update_checker import _REPO, _read_token

_PROJECT_ROOT = Path(__file__).parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_MODELS_TAG = "models-v1"
_RELEASE_URL = f"https://api.github.com/repos/{_REPO}/releases/tags/{_MODELS_TAG}"

# The release also carries optional sample-specific models. Only these are needed
# for NucLogic to run, so only these are fetched automatically; the rest are left
# for the user to download from the release page and drop into models/.
REQUIRED_ASSETS = {
    "2d_high_quality_model",
    "2d_low_quality_model",
    "sam2.1_hiera_small.pt",
}


def _request(url, accept):
    req = urllib.request.Request(url, headers={"Accept": accept})
    token = _read_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def _fetch_assets():
    with urllib.request.urlopen(
        _request(_RELEASE_URL, "application/vnd.github+json"), timeout=30
    ) as resp:
        return json.loads(resp.read()).get("assets", [])


def optional_assets():
    """The extra models, each tagged with whether it is already downloaded."""
    extras = []
    for asset in _fetch_assets():
        if asset["name"] in REQUIRED_ASSETS:
            continue
        local = _MODELS_DIR / asset["name"]
        asset["installed"] = local.exists() and local.stat().st_size == asset["size"]
        extras.append(asset)
    return extras


def missing_assets():
    """Release assets whose local copy is absent or the wrong size.

    Size is compared against the release rather than a hardcoded number, so
    re-uploading a model is picked up without touching this file.
    """
    assets = _fetch_assets()

    return [
        a
        for a in assets
        if a["name"] in REQUIRED_ASSETS
        and (
            not (_MODELS_DIR / a["name"]).exists()
            or (_MODELS_DIR / a["name"]).stat().st_size != a["size"]
        )
    ]


def have_required_locally():
    return all((_MODELS_DIR / name).exists() for name in REQUIRED_ASSETS)


def download_asset(asset, progress=None):
    """Download one asset into models/. progress(done_bytes, total_bytes) if given.

    Written to .part and renamed only once complete, so an interrupted download
    cannot leave a truncated file that the size check would later accept.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = _MODELS_DIR / asset["name"]
    part = destination.with_name(destination.name + ".part")

    # octet-stream (not the JSON accept) is what makes the API return the file
    # itself; it redirects to a CDN URL and urllib carries the auth header across.
    with urllib.request.urlopen(
        _request(asset["url"], "application/octet-stream"), timeout=120
    ) as resp, open(part, "wb") as f:
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, asset["size"])

    part.replace(destination)


class ModelDownloader(QThread):
    # "completed" rather than "finished": QThread already defines finished.
    progress = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(self, assets=None, parent=None):
        # assets=None means "work out which required models are missing"; a list
        # means download exactly those (used by the extra-models picker).
        super().__init__(parent)
        self._assets = assets

    def run(self):
        if self._assets is not None:
            missing = self._assets
        else:
            try:
                missing = missing_assets()
            except Exception as e:
                # Offline is only fatal when the weights are not already on disk
                # -- an installed copy must still start without reaching GitHub.
                if have_required_locally():
                    self.completed.emit()
                else:
                    self.error.emit(str(e))
                return

        try:
            for i, asset in enumerate(missing, 1):
                def report(done, total, name=asset["name"], i=i):
                    self.progress.emit(
                        f"Downloading {name} ({i}/{len(missing)}) — "
                        f"{done / 1048576:.0f} of {total / 1048576:.0f} MB"
                    )

                download_asset(asset, progress=report)
            self.completed.emit()
        except Exception as e:
            self.error.emit(str(e))
