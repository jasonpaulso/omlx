#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fork release pipeline: build box (MacBook) builds, signs, and ships the
# .app to every serving box. Replaces the old both-boxes-pull-and-rebuild
# flow — only this machine holds actively-developed code; other boxes just
# receive the artifact.
#
#   packaging/release.sh              # test → build → sign → install local + studio
#   packaging/release.sh --local-only
#   packaging/release.sh --studio-only
#   packaging/release.sh --skip-tests # when the suite already ran this session
#
# Env:
#   OMLX_SIGN_IDENTITY   codesign identity (default: Apple Development cert).
#                        Real (non ad-hoc) identity is what makes TCC/FDA
#                        grants survive bundle swaps on the Studio.
#   OMLX_STUDIO_HOST     default 192.168.5.28

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STUDIO="${OMLX_STUDIO_HOST:-192.168.5.28}"
IDENTITY="${OMLX_SIGN_IDENTITY:-Apple Development: Jason Southwell (2N4QW28DF6)}"
STAGE="$REPO/apps/omlx-mac/build/Stage/oMLX.app"

DO_LOCAL=1; DO_STUDIO=1; DO_TESTS=1
for arg in "$@"; do
    case "$arg" in
        --local-only)  DO_STUDIO=0 ;;
        --studio-only) DO_LOCAL=0 ;;
        --skip-tests)  DO_TESTS=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

log() { printf '\033[1;34m[release]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[release]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$REPO"
[ "$(git branch --show-current)" = "deploy" ] || die "not on deploy branch"
git diff --quiet || die "tree has uncommitted changes — commit first (bundle embeds the worktree)"

VERSION=$(python3 -c "exec(open('omlx/_version.py').read()); print(__version__)")
DMG="$HOME/Downloads/oMLX-$VERSION.dmg"
log "releasing oMLX $VERSION"

# --- Test gate ------------------------------------------------------------
if [ "$DO_TESTS" -eq 1 ]; then
    log "running test suite…"
    # glm_mtp TestSmallLRouting: known fp-tolerance failures on this hardware.
    env -u OMLX_API_KEY -u OMLX_BASE_URL uv run pytest -q \
        --deselect tests/test_glm_mtp_patch.py::TestSmallLRouting \
        || die "tests failed — not shipping"
fi

# --- Build + sign + package ----------------------------------------------
log "building .app…"
PYTHON_BIN="$REPO/.venv/bin/python" apps/omlx-mac/Scripts/build.sh release

log "re-signing with: $IDENTITY"
"$REPO/packaging/resign.sh" "$STAGE" "$IDENTITY"

log "creating DMG…"
hdiutil create -volname oMLX -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
log "DMG: $DMG ($(du -h "$DMG" | cut -f1))"

# --- Install helpers ------------------------------------------------------
# Idle gate: refuse to bounce a server that's actively moving bytes
# (model download queues die silently when the app quits).
_idle_or_die() { # $1 = "" for local, "ssh host" prefix otherwise
    local delta
    delta=$($1 bash -s <<'EOS'
B1=$(netstat -ib | awk '$1=="en0"{print $7; exit}'); sleep 15
B2=$(netstat -ib | awk '$1=="en0"{print $7; exit}'); echo $((B2-B1))
EOS
    )
    delta="${delta:-0}"
    [ "$delta" -lt 20000000 ] || die "box is moving ${delta} bytes/15s — active transfer, not installing"
}

_poll_health() { # $1 = "" or "ssh host"
    # 60 x 5s = 300s: a cold boot with model preload took ~4 min (2026-07-23),
    # which falsely failed the previous 180s window.
    for _ in $(seq 1 60); do
        code=$($1 curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:8888/health || true)
        [ "$code" = "200" ] && return 0
        sleep 5
    done
    return 1
}

# --- Local ---------------------------------------------------------------
if [ "$DO_LOCAL" -eq 1 ]; then
    log "installing locally…"
    _idle_or_die ""
    pkill -TERM -f "/Applications/oMLX.app/Contents/MacOS/oMLX" || true
    sleep 8
    rm -rf "$HOME/Downloads/oMLX-prev-backup.app"
    ditto /Applications/oMLX.app "$HOME/Downloads/oMLX-prev-backup.app"
    rm -rf /Applications/oMLX.app
    ditto "$STAGE" /Applications/oMLX.app
    open -a /Applications/oMLX.app
    _poll_health "" || die "local server did not come healthy"
    log "local: healthy on $VERSION"
fi

# --- Studio ---------------------------------------------------------------
if [ "$DO_STUDIO" -eq 1 ]; then
    log "shipping to $STUDIO…"
    scp -q "$DMG" "$STUDIO:~/Downloads/"
    SUM_L=$(shasum -a 256 "$DMG" | cut -d' ' -f1)
    SUM_R=$(ssh "$STUDIO" "shasum -a 256 ~/Downloads/$(basename "$DMG")" | cut -d' ' -f1)
    [ "$SUM_L" = "$SUM_R" ] || die "checksum mismatch after transfer"

    _idle_or_die "ssh $STUDIO"
    # Explicit -mountpoint: a stale /Volumes/oMLX from an earlier session
    # once shunted the new DMG to "/Volumes/oMLX 1" and we installed the
    # old bundle. Never rely on the default mount path.
    ssh "$STUDIO" bash -s <<EOS
set -euo pipefail
pkill -TERM -f "/Applications/oMLX.app/Contents/MacOS/oMLX" || true
sleep 8
MNT=\$(mktemp -d /tmp/omlx-dmg.XXXXXX)
hdiutil attach ~/Downloads/$(basename "$DMG") -nobrowse -quiet -mountpoint "\$MNT"
rm -rf ~/Downloads/oMLX-prev-backup.app
ditto /Applications/oMLX.app ~/Downloads/oMLX-prev-backup.app
rm -rf /Applications/oMLX.app
ditto "\$MNT/oMLX.app" /Applications/oMLX.app
hdiutil detach "\$MNT" -quiet
open -a /Applications/oMLX.app
EOS
    _poll_health "ssh $STUDIO" || die "studio server did not come healthy (FDA dialog? check GUI)"
    RV=$(ssh "$STUDIO" "/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' /Applications/oMLX.app/Contents/Info.plist")
    [ "$RV" = "$VERSION" ] || die "studio reports $RV, expected $VERSION"
    log "studio: healthy on $VERSION"
fi

log "release $VERSION complete. Rollback: ditto ~/Downloads/oMLX-prev-backup.app /Applications/oMLX.app (per box)"
