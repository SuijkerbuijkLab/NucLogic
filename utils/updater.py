from pathlib import Path
import subprocess
import tempfile
import shutil
import zipfile
import urllib.request

from PySide6.QtCore import QThread, Signal

_PROJECT_ROOT = Path(__file__).parent.parent


def _read_token():
    p = _PROJECT_ROOT / "github_token.txt"
    return p.read_text().strip() if p.exists() else None


# Files/folders that must never be overwritten by an update
_PRESERVE = {".pixi", "tools", "github_token.txt", "logs", "models"}


class Updater(QThread):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, zipball_url, version=None, parent=None):
        super().__init__(parent)
        self._url = zipball_url
        self._version = version  # release tag being installed; written to version.txt

    def run(self):
        try:
            tmp = Path(tempfile.mkdtemp(prefix="nuclogic_update_"))
            zip_path = tmp / "release.zip"

            # --- Download ---
            self.progress.emit("Downloading update…")
            req = urllib.request.Request(self._url, headers={"Accept": "application/vnd.github+json"})
            token = _read_token()
            if token:
                req.add_header("Authorization", f"token {token}")
            with urllib.request.urlopen(req, timeout=120) as resp:
                zip_path.write_bytes(resp.read())

            # --- Extract ---
            self.progress.emit("Extracting files…")
            extract_dir = tmp / "extracted"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # GitHub wraps everything in a top-level folder — find it
            children = list(extract_dir.iterdir())
            source_root = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

            # --- Copy over project root, skipping preserved paths ---
            self.progress.emit("Applying update…")
            for item in source_root.iterdir():
                if item.name in _PRESERVE:
                    continue
                dest = _PROJECT_ROOT / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            # --- Detect whether pixi.toml changed ---
            new_pixi_toml = source_root / "pixi.toml"
            current_pixi_toml = _PROJECT_ROOT / "pixi.toml"
            pixi_toml_changed = (
                new_pixi_toml.exists()
                and (
                    not current_pixi_toml.exists()
                    or new_pixi_toml.read_bytes() != current_pixi_toml.read_bytes()
                )
            )

            if pixi_toml_changed:
                # Wipe the old environment so pixi rebuilds cleanly.
                # Stale packages (e.g. pyside2 after a pyside6 migration) would
                # otherwise remain and cause crashes.
                self.progress.emit("Dependencies changed — clearing old environment…")
                envs_dir = _PROJECT_ROOT / ".pixi" / "envs"
                if envs_dir.exists():
                    shutil.rmtree(envs_dir, ignore_errors=True)
                self.progress.emit("Rebuilding environment — this may take a few minutes…")
            else:
                self.progress.emit("Syncing dependencies (pixi install)…")

            pixi_exe = _PROJECT_ROOT / "tools" / "pixi.exe"
            if not pixi_exe.exists():
                pixi_exe = "pixi"  # fall back to PATH
            result = subprocess.run(
                [str(pixi_exe), "install"],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pixi install failed:\n{result.stderr}")

            # Record the installed version locally. version.txt is not part of the
            # release archive (it is gitignored), so we write it from the tag we
            # just installed — this is what stops the update prompt from recurring.
            if self._version:
                from utils.update_checker import _write_local_version

                _write_local_version(self._version)

            shutil.rmtree(tmp, ignore_errors=True)
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
