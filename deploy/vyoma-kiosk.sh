#!/usr/bin/env bash
# Kiosk launcher for the Vyoma dashboard.
#
# Runs at GNOME login (via ~/.config/autostart/vyoma-kiosk.desktop). The
# vyoma-receptionist *user service* serves the dashboard on port 7860; this
# script just waits for that port to answer, then opens Chrome fullscreen
# pointing at it. A dedicated profile dir keeps the "restore pages?" prompt
# and any personal Chrome profile out of the kiosk.
set -u

# The robot's own screen shows the visitor-facing avatar face; the
# operator dashboard stays reachable over the LAN at http://<robot>:7860/
# (set URL="$BASE/" to show the dashboard on the robot instead).
BASE="http://localhost:7860"
URL="$BASE/avatar"
PROFILE_DIR="$HOME/.local/share/vyoma-kiosk-profile"

# Wait (up to ~60s) for the dashboard to start serving before opening it.
for _ in $(seq 1 60); do
  if curl -sf -o /dev/null "$BASE/"; then
    break
  fi
  sleep 1
done

# Clear the exit flags so Chrome never shows the crash/restore bar after a
# hard power-off of the robot.
if [ -f "$PROFILE_DIR/Default/Preferences" ]; then
  sed -i 's/"exit_type":"[^"]*"/"exit_type":"Normal"/; s/"exited_cleanly":false/"exited_cleanly":true/' \
    "$PROFILE_DIR/Default/Preferences" 2>/dev/null || true
fi

exec google-chrome \
  --kiosk \
  --app="$URL" \
  --user-data-dir="$PROFILE_DIR" \
  --no-first-run \
  --no-default-browser-check \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-features=TranslateUI \
  --password-store=basic \
  --start-fullscreen
