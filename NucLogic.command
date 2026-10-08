#!/bin/bash
# NucLogic launcher for macOS: double-click this file in Finder to start NucLogic.
# Finder opens .command files in Terminal; this runs the same launcher script as
# Linux, which sets up the environment on first run and then starts the app.
cd "$(dirname "$0")" || exit 1
exec bash ./Linux_NucLogic.sh "$@"
