#!/usr/bin/env bash

runid="$1"
if [ -z "$runid" ]; then
  echo "Usage: $0 <run_id>"
  exit 1
fi

set -e
source ~/asparagus/.env
set +e

if [ -z "$ASPARAGUS_MODELS" ]; then
  echo "Error: ASPARAGUS_MODELS environment variable is not set."
  exit 1
fi

find $ASPARAGUS_MODELS -type d -name "run_id=${runid}" -print -quit
