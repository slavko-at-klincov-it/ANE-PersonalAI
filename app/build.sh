#!/bin/bash
# build.sh — Build Personal AI menu bar app
set -e

cd "$(dirname "$0")"

echo "=== Building Personal AI ==="
swift build -c release 2>&1

# Create .app bundle
APP="PersonalAI.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"

cp .build/release/PersonalAI "$APP/Contents/MacOS/"
cp Resources/Info.plist "$APP/Contents/"

echo ""
echo "=== Build complete: $APP ==="
echo ""
echo "To run:     open $APP"
echo "To install: cp -r $APP /Applications/"
