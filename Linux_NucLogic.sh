#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI_LOCAL="$SCRIPT_DIR/tools/pixi"

# Keep the package cache in the install folder, matching NucLogic.bat, so an
# environment rebuild reuses local packages instead of re-downloading them.
export PIXI_CACHE_DIR="$SCRIPT_DIR/pixi_cache"

if [ -f "$PIXI_LOCAL" ]; then
    PIXI="$PIXI_LOCAL"
elif command -v pixi &>/dev/null; then
    PIXI="pixi"
else
    echo "pixi not found. Downloading into project tools folder..."
    mkdir -p "$SCRIPT_DIR/tools"
    if command -v curl &>/dev/null; then
        curl -fsSL "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl" -o "$PIXI_LOCAL"
    elif command -v wget &>/dev/null; then
        wget -q "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl" -O "$PIXI_LOCAL"
    else
        echo "ERROR: Neither curl nor wget found. Install pixi manually from https://pixi.sh"
        exit 1
    fi
    chmod +x "$PIXI_LOCAL"
    PIXI="$PIXI_LOCAL"
fi

cd "$SCRIPT_DIR"
echo "Starting NucLogic via pixi..."
"$PIXI" run start
