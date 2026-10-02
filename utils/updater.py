from pathlib import Path
import os
import subprocess
import tempfile
import time
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

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _dependency_names(pixi_toml):
    import tomllib

    with open(pixi_toml, "rb") as f:
        data = tomllib.load(f)
    return set(data.get("dependencies", {})) | set(data.get("pypi-dependencies", {}))


def _dependencies_removed(new_toml, current_toml):
    """True when the update drops or renames a package.

    Only then can stale files be left behind that a plain `pixi install` will
    not clear. Added packages need no rebuild.
    """
    if not new_toml.exists() or not current_toml.exists():
        return False
    try:
        return bool(_dependency_names(current_toml) - _dependency_names(new_toml))
    except Exception:
        return True


# Above this many changed files, per-file icacls calls cost more than one
# recursive pass (a full environment rebuild lands ~70k files).
_BULK_ACL_THRESHOLD = 2000


def _iter_files(root):
    """Walk `root` with scandir, yielding DirEntry for each file.

    On Windows the directory read already carries the metadata, so
    entry.stat() below costs no extra syscall -- roughly an order of magnitude
    faster than pathlib.rglob + stat over an environment's ~70k files.
    Unreadable directories are skipped rather than aborting the walk.
    """
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        yield entry
        except OSError:
            continue


def _grant_new_files_access(root, since):
    r"""Re-apply access to just the files this install created.

    _grant_all_users_access walks the whole tree twice, which is ~225k file
    operations and takes minutes on every update even when pixi installed
    nothing. Only files freshly linked in by this install can carry a cache
    DACL, so only those need fixing.

    Scoped to .pixi: a hardlink shares one inode with its pixi_cache source, so
    resetting the env copy already fixes the cache copy.

    Packages already present in the cache keep their old mtime and are skipped;
    those were granted by an earlier run or by NucLogic.bat's first-time setup.
    """
    if not root.exists():
        return

    try:
        changed = [e.path for e in _iter_files(root) if e.stat().st_mtime > since]
    except OSError:
        _grant_all_users_access(root)
        return

    if not changed:
        return
    if len(changed) > _BULK_ACL_THRESHOLD:
        _grant_all_users_access(root)
        return

    for path in changed:
        try:
            result = subprocess.run(
                ["icacls", str(path), "/reset", "/C", "/Q"],
                capture_output=True,
                text=True,
            )
            # takeown is only needed when the file belongs to another account,
            # so it is paid for on failure rather than up front.
            if result.returncode != 0:
                subprocess.run(
                    ["takeown", "/F", str(path)], capture_output=True, text=True
                )
                subprocess.run(
                    ["icacls", str(path), "/reset", "/C", "/Q"],
                    capture_output=True,
                    text=True,
                )
        except Exception:
            pass


def _grant_all_users_access(root):
    r"""Re-apply inheritable full access for BUILTIN\Users across a whole tree.

    The fallback for _grant_new_files_access: used for a full rebuild, or when
    the tree cannot be scanned.

    pixi/rattler hardlinks package files from PIXI_CACHE_DIR into .pixi\envs, and a
    hardlink carries the cache file's own DACL rather than inheriting the install
    folder's ACL. Packages freshly downloaded during an update therefore land
    without a Users ACE, so standard accounts hit "Access is denied" loading their
    DLLs -- the one-time icacls grant in NucLogic.bat only covered files that
    existed at first-time setup. Re-granting here fixes every account.

    Best-effort. takeown lets us rewrite the DACL even on files owned by another
    account. icacls /reset then discards each file's explicit ACL and forces it to
    re-inherit the install folder's (permissive) ACL -- this is what clears the
    protected, inheritance-disabled DACLs that pip/uv package files land with (a
    plain /grant would only append an entry and leave the protected flag set). /T
    recurses, /C keeps going past briefly locked files, /Q hides success spam. Any
    failure is non-fatal -- the install itself already succeeded.
    """
    root = str(root)
    for cmd in (
        ["takeown", "/F", root, "/R", "/D", "Y"],
        ["icacls", root, "/reset", "/T", "/C", "/Q"],
    ):
        try:
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            pass


def _is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _stash_lfs_pointer_targets(source_root, stash_dir):
    """Move aside local files that the release would replace with LFS pointers.

    GitHub zipballs carry the pointer text rather than the tracked content, so
    copying them over a real file destroys it.
    """
    stashed = []
    for item in source_root.rglob("*"):
        relative = item.relative_to(source_root)
        if relative.parts[0] in _PRESERVE or not item.is_file():
            continue
        if not _is_lfs_pointer(item):
            continue

        local = _PROJECT_ROOT / relative
        if not local.is_file() or _is_lfs_pointer(local):
            continue

        target = stash_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local), str(target))
        stashed.append(relative)
    return stashed


def _restore_lfs_pointer_targets(stashed, stash_dir):
    for relative in stashed:
        destination = _PROJECT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(stash_dir / relative), str(destination))


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

            # Decide this before the copy below overwrites the current pixi.toml.
            needs_rebuild = _dependencies_removed(
                source_root / "pixi.toml", _PROJECT_ROOT / "pixi.toml"
            )

            stash_dir = tmp / "preserved"
            stashed = _stash_lfs_pointer_targets(source_root, stash_dir)

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

            _restore_lfs_pointer_targets(stashed, stash_dir)

            if needs_rebuild:
                self.progress.emit("Dependencies changed — clearing old environment…")
                envs_dir = _PROJECT_ROOT / ".pixi" / "envs"
                if envs_dir.exists():
                    shutil.rmtree(envs_dir, ignore_errors=True)
                self.progress.emit("Rebuilding environment — this may take a few minutes…")
            else:
                self.progress.emit("Syncing dependencies (pixi install)…")

            pixi_exe = _PROJECT_ROOT / "tools" / ("pixi.exe" if os.name == "nt" else "pixi")
            if not pixi_exe.exists():
                pixi_exe = "pixi"  # fall back to PATH
            # Timestamped so the permission step below can tell which files this
            # install actually produced. Taken before the run, and nudged back a
            # second to absorb filesystem timestamp granularity.
            install_started = time.time() - 1
            result = subprocess.run(
                [str(pixi_exe), "install"],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pixi install failed:\n{result.stderr}")

            # Packages freshly hardlinked from the cache during this install do not
            # inherit the install folder's ACLs, so re-apply them or standard
            # accounts get "Access is denied" loading the new DLLs.
            self.progress.emit("Updating file permissions for all users…")
            _grant_new_files_access(_PROJECT_ROOT / ".pixi", install_started)

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
