#!/bin/bash

SAVE_DIR="data"
SAVE_FILE="polygon.zip"

python3 script/download_data.py \
--save-dir "$SAVE_DIR" \
--save-file "$SAVE_FILE"

cd "$SAVE_DIR" && unzip "$SAVE_FILE"
