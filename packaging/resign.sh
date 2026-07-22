#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fork-owned post-build re-sign: replaces the ad-hoc signatures build.sh
# applies with a real (Apple-issued) identity so the bundle's designated
# requirement becomes team-based and TCC grants (Full Disk Access) survive
# bundle swaps across deploys.
#
# Deliberately separate from apps/omlx-mac/Scripts/build.sh, which is
# upstream-owned (merge-seam rule: no fork lines in upstream files).
#
# Usage:
#   packaging/resign.sh /path/to/oMLX.app "Apple Development: Jason Southwell (2N4QW28DF6)"
#
# No --timestamp (not notarizing; keeps ~800 Mach-O signs fast) and no
# hardened runtime (would break MLX JIT / unsigned-memory without extra
# entitlements, and is only needed for notarization).

set -euo pipefail

APP="${1:?usage: resign.sh /path/to/oMLX.app <identity>}"
IDENTITY="${2:?usage: resign.sh /path/to/oMLX.app <identity>}"

[ -d "$APP" ] || { echo "error: $APP is not a directory" >&2; exit 1; }

PYTHON_DIR="$APP/Contents/Resources/Python"
CLI_WRAPPER="$APP/Contents/MacOS/omlx-cli"

# Same embedded Mach-O walk as build.sh's _sign_embedded_mach_o_files:
# .so/.dylib/.bundle plus anything executable, skipping dSYM/__pycache__.
count=0
while IFS= read -r -d '' path; do
    file -b "$path" 2>/dev/null | grep -q "Mach-O" || continue
    codesign --force --sign "$IDENTITY" "$path" >/dev/null 2>&1
    count=$((count + 1))
done < <(
    find "$PYTHON_DIR" \
        \( -path "*/.dSYM/*" -o -path "*/__pycache__/*" \) -prune -o \
        -type f \( \
            -name "*.so" -o \
            -name "*.dylib" -o \
            -name "*.bundle" -o \
            -perm -100 -o \
            -perm -010 -o \
            -perm -001 \
        \) -print0
)
echo "re-signed $count embedded Mach-O files"

[ -f "$CLI_WRAPPER" ] && codesign --force --sign "$IDENTITY" "$CLI_WRAPPER" >/dev/null 2>&1 \
    && echo "re-signed omlx-cli wrapper"

codesign --force --sign "$IDENTITY" "$APP"
echo "re-signed app bundle"

# Show the designated requirement — this is what TCC matches against.
codesign -d -r- "$APP" 2>&1 | grep "designated"

# No --strict: the venvstacks Python layer ships ~100 broken symlinks in
# share/venv/dynlib (links into site-packages of a different layer), which
# --strict rejects. They're pre-existing in every deployed bundle and inert
# at runtime. spctl will also reject (dev cert, not notarized) — irrelevant
# for scp/CLI deploys, which carry no quarantine attribute.
codesign --verify "$APP" && echo "verify: OK"
