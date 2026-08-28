"""Exercise the updater helpers against a simulated release tree."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(PROJECT_ROOT)
sys.path.insert(0, str(ROOT))

# Import without pulling in PySide6 (not needed for the helpers under test).
import types

fake_qtcore = types.ModuleType("PySide6.QtCore")
fake_qtcore.QThread = type("QThread", (), {"__init__": lambda self, *a, **k: None})
fake_qtcore.Signal = lambda *a, **k: None
fake_pyside = types.ModuleType("PySide6")
fake_pyside.QtCore = fake_qtcore
sys.modules.setdefault("PySide6", fake_pyside)
sys.modules.setdefault("PySide6.QtCore", fake_qtcore)

import utils.updater as up

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:8eb7b67a11\nsize 1848392\n"
)

BASE_TOML = """
[dependencies]
python = "3.11.*"
numpy = "*"

[pypi-dependencies]
torch = ">=2.5.1"
cellpose = "*"
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


print("\n== _dependencies_removed ==")
tmp = Path(tempfile.mkdtemp())
current = tmp / "current.toml"
write(current, BASE_TOML)

added = tmp / "added.toml"
write(added, BASE_TOML + '\nnd2 = "*"\nbioio = "*"\n')
check("pure addition -> no rebuild", up._dependencies_removed(added, current) is False)

removed = tmp / "removed.toml"
write(removed, BASE_TOML.replace('cellpose = "*"', ""))
check("removal -> rebuild", up._dependencies_removed(removed, current) is True)

renamed = tmp / "renamed.toml"
write(renamed, BASE_TOML.replace("pyside2", "x").replace('numpy = "*"', 'numpy2 = "*"'))
check("rename -> rebuild", up._dependencies_removed(renamed, current) is True)

comment_only = tmp / "comment.toml"
write(comment_only, BASE_TOML + "\n# a comment, version bump, whatever\n")
check("comment/version churn -> no rebuild",
      up._dependencies_removed(comment_only, current) is False)

broken = tmp / "broken.toml"
write(broken, "this is not valid toml [[[")
check("unparseable -> rebuild (conservative)",
      up._dependencies_removed(broken, current) is True)

check("missing current -> no rebuild",
      up._dependencies_removed(added, tmp / "nope.toml") is False)

print("\n== _is_lfs_pointer ==")
p = tmp / "pointer.mp4"
p.write_bytes(POINTER)
check("detects pointer", up._is_lfs_pointer(p) is True)
real = tmp / "real.mp4"
real.write_bytes(b"\x00\x01\x02" * 5000)
check("detects real file", up._is_lfs_pointer(real) is False)
check("missing file is not pointer", up._is_lfs_pointer(tmp / "gone.mp4") is False)

print("\n== stash/restore against a simulated update ==")
project = Path(tempfile.mkdtemp()) / "NucLogic"
release = Path(tempfile.mkdtemp()) / "release"
stash = Path(tempfile.mkdtemp()) / "preserved"
up._PROJECT_ROOT = project

# Local install: real LFS content, real models, plus a normal file.
write(project / "PySide6" / "app.py", "old app\n")
(project / "miscellaneous").mkdir(parents=True)
(project / "miscellaneous" / "example_movie.mp4").write_bytes(b"\xff\xd8REAL VIDEO" * 900)
write(project / "miscellaneous" / "environment.yml", "old env\n")
(project / "models").mkdir(parents=True)
(project / "models" / "2d_high_quality_model").write_bytes(b"REAL WEIGHTS" * 1000)
write(project / "README.md", "old readme\n")

# Release zipball: pointers where LFS content belongs.
write(release / "PySide6" / "app.py", "new app\n")
(release / "miscellaneous").mkdir(parents=True)
(release / "miscellaneous" / "example_movie.mp4").write_bytes(POINTER)
write(release / "miscellaneous" / "environment.yml", "new env\n")
(release / "models").mkdir(parents=True)
(release / "models" / "2d_high_quality_model").write_bytes(POINTER)
write(release / "README.md", "new readme\n")

video_before = (project / "miscellaneous" / "example_movie.mp4").read_bytes()
models_before = (project / "models" / "2d_high_quality_model").read_bytes()

stashed = up._stash_lfs_pointer_targets(release, stash)
check("stashes only the unprotected LFS file",
      [str(s).replace("\\", "/") for s in stashed] == ["miscellaneous/example_movie.mp4"],
      f"got {stashed}")
check("models/ never touched (still on disk, full size)",
      (project / "models" / "2d_high_quality_model").read_bytes() == models_before)

# Replicate the updater's copy loop.
for item in release.iterdir():
    if item.name in up._PRESERVE:
        continue
    dest = project / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

check("copy would have clobbered the video",
      (project / "miscellaneous" / "example_movie.mp4").read_bytes() == POINTER)

up._restore_lfs_pointer_targets(stashed, stash)

check("video restored to real content",
      (project / "miscellaneous" / "example_movie.mp4").read_bytes() == video_before)
check("normal files still updated",
      (project / "PySide6" / "app.py").read_text() == "new app\n")
check("sibling file in same dir updated",
      (project / "miscellaneous" / "environment.yml").read_text() == "new env\n")
check("root file updated", (project / "README.md").read_text() == "new readme\n")
check("models untouched by whole flow",
      (project / "models" / "2d_high_quality_model").read_bytes() == models_before)

print("\n== no LFS files in release: no-op ==")
project2 = Path(tempfile.mkdtemp()) / "NucLogic2"
release2 = Path(tempfile.mkdtemp()) / "release2"
stash2 = Path(tempfile.mkdtemp()) / "preserved2"
up._PROJECT_ROOT = project2
write(project2 / "a.txt", "local\n")
write(release2 / "a.txt", "new\n")
check("nothing stashed", up._stash_lfs_pointer_targets(release2, stash2) == [])

print("\n== local file already a pointer: left alone ==")
project3 = Path(tempfile.mkdtemp()) / "NucLogic3"
release3 = Path(tempfile.mkdtemp()) / "release3"
stash3 = Path(tempfile.mkdtemp()) / "preserved3"
up._PROJECT_ROOT = project3
(project3 / "miscellaneous").mkdir(parents=True)
(project3 / "miscellaneous" / "example_movie.mp4").write_bytes(POINTER)
(release3 / "miscellaneous").mkdir(parents=True)
(release3 / "miscellaneous" / "example_movie.mp4").write_bytes(POINTER)
check("pointer-vs-pointer not stashed",
      up._stash_lfs_pointer_targets(release3, stash3) == [])

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
