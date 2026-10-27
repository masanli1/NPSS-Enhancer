#!/bin/bash
cd "$(dirname "$0")/GUI" || exit 1
exec ./run_gui.command
