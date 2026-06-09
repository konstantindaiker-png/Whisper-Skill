#!/bin/bash
# Push-to-talk voice dictation launcher
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"
exec .venv/bin/python3 -u -m examples.voice_dictation "$@"
