#!/bin/bash
# NucLogic launcher for Linux, mirroring NucLogic.bat.
#
# There is no administrator step here: pixi writes only inside this folder, and
# sharing an install between Linux accounts is done with ordinary group
# permissions rather than the elevated ACL grant the Windows launcher offers.
#
# "Linux_NucLogic.sh --setup" builds the environment and exits without launching,
# e.g. to prepare a shared install on a server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Keep the package cache in the install folder, matching NucLogic.bat. It has to
# sit on the same filesystem as the environment for pixi's hardlink step, and lets
# a rebuild reuse local packages instead of re-downloading them.
export PIXI_CACHE_DIR="$SCRIPT_DIR/pixi_cache"
ENV_DIR="$SCRIPT_DIR/.pixi/envs/default"
PIXI_LOCAL="$SCRIPT_DIR/tools/pixi"
PIXI_URL="https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl"

SETUP_ONLY=""
[ "$1" = "--setup" ] && SETUP_ONLY=1

die() {
    echo
    echo "ERROR: $*"
    # Keep the window open when started by double-clicking, as NucLogic.bat does.
    if [ -t 0 ]; then
        read -r -p "Press Enter to close..."
    fi
    exit 1
}

# Locate pixi, downloading it if needed, and confirm the binary actually runs.
ensure_pixi() {
    if [ -f "$PIXI_LOCAL" ]; then
        chmod +x "$PIXI_LOCAL" 2>/dev/null
        PIXI="$PIXI_LOCAL"
    elif command -v pixi >/dev/null 2>&1; then
        PIXI="pixi"
    else
        if [ "$(uname -m)" != "x86_64" ]; then
            die "NucLogic supports 64-bit x86 Linux only (this machine is $(uname -m))."
        fi
        echo "pixi not found. Downloading it into the tools folder..."
        mkdir -p "$SCRIPT_DIR/tools" || die "could not create $SCRIPT_DIR/tools; this folder is not writable by your account."

        # Download to a temporary name so an interrupted download never leaves a
        # half-written "pixi" that later runs would try to use.
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$PIXI_URL" -o "$PIXI_LOCAL.part"
        elif command -v wget >/dev/null 2>&1; then
            wget -q "$PIXI_URL" -O "$PIXI_LOCAL.part"
        else
            die "neither curl nor wget is installed. Install one of them, or install pixi yourself from https://pixi.sh, and run this script again."
        fi || {
            rm -f "$PIXI_LOCAL.part"
            die "could not download pixi. Common causes:
  - no internet connection, or a proxy blocking github.com
  - this folder is not writable by your account
Alternatively install pixi yourself from https://pixi.sh and run this script again."
        }
        mv "$PIXI_LOCAL.part" "$PIXI_LOCAL"
        chmod +x "$PIXI_LOCAL"
        PIXI="$PIXI_LOCAL"
    fi

    # A captive portal or proxy can answer with an HTML page and HTTP 200, leaving
    # a "pixi" that is not an executable. Catch that here rather than letting it
    # fail confusingly inside pixi install.
    if ! "$PIXI" --version >/dev/null 2>&1; then
        if [ "$PIXI" = "$PIXI_LOCAL" ]; then
            rm -f "$PIXI_LOCAL"
            die "the downloaded pixi does not run and has been deleted. Run this script again to download it afresh."
        fi
        die "the pixi on your PATH does not run. Reinstall it from https://pixi.sh."
    fi
}

ensure_pixi

# ----------------------------------------------------------------- first run ---
if [ ! -d "$ENV_DIR" ]; then
    echo "First-time setup: building the environment, this may take several minutes..."
    "$PIXI" install || die "pixi install failed with exit code $?.
If the message above mentions git, install git (for example 'sudo apt install git')
and run this script again."
fi

if [ -n "$SETUP_ONLY" ]; then
    echo "Setup complete."
    exit 0
fi

# -------------------------------------------------------------------- launch ---
echo "Starting NucLogic via pixi..."
"$PIXI" run start || die "pixi run start failed with exit code $?."
