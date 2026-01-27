#!/usr/bin/env bash

set -euo pipefail

PACKAGE="$1"

# Color codes
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

pip install --upgrade pip > /dev/null

VERSIONS=$(pip index versions "$PACKAGE" 2>/dev/null | grep -oP '\d+(\.\d+)+([a-zA-Z0-9]*)?' | sort -Vr)

SUCCESSFUL=()

for VERSION in $VERSIONS; do
    if pip install "$PACKAGE==$VERSION"; then
        echo -e "${GREEN}${PACKAGE}==${VERSION} SUCCESS${NC}"
        SUCCESSFUL+=("$VERSION")
    else
        echo -e "${YELLOW}${PACKAGE}==${VERSION} MAYBE FAIL${NC}"
    fi
done

if [[ ${#SUCCESSFUL[@]} -eq 0 ]]; then
    echo -e "${YELLOW}Maybe no installable versions found for $PACKAGE$, please read output carefully${NC}"
    exit 1
fi