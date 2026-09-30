#!/usr/bin/env bash
# Builds the release binary, then a tarball and an AppImage in dist/.
# Run scripts/fetch-llama.sh first so vendor/llama holds the server.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --release
[ -x vendor/llama/llama-server ] || { echo "vendor/llama is empty. Run scripts/fetch-llama.sh first." >&2; exit 1; }
version="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
mkdir -p dist

# Tarball: the binary with the server in a llama folder beside it, where Scoobert looks for it.
tarname="scoobert-$version-linux-x86_64"
rm -rf "dist/$tarname"
mkdir -p "dist/$tarname/llama"
cp target/release/scoobert LICENSE "dist/$tarname/"
cp -P vendor/llama/* "dist/$tarname/llama/"
cp packaging/linux/scoobert.desktop assets/icon.png packaging/portable.txt "dist/$tarname/"
tar -C dist -czf "dist/$tarname.tar.gz" "$tarname"

# AppImage: the server goes in usr/lib/scoobert/llama, which Scoobert also checks.
app=dist/AppDir
rm -rf "$app"
mkdir -p "$app/usr/bin" "$app/usr/lib/scoobert/llama" "$app/usr/share/applications" "$app/usr/share/icons/hicolor/256x256/apps"
cp target/release/scoobert "$app/usr/bin/"
cp -P vendor/llama/* "$app/usr/lib/scoobert/llama/"
cp packaging/linux/scoobert.desktop "$app/usr/share/applications/"
cp packaging/linux/scoobert.desktop "$app/"
cp assets/icon.png "$app/usr/share/icons/hicolor/256x256/apps/scoobert.png"
cp assets/icon.png "$app/scoobert.png"
cp packaging/linux/AppRun "$app/AppRun"
chmod +x "$app/AppRun" "$app/usr/bin/scoobert"

tool="${APPIMAGETOOL:-appimagetool}"
if ! command -v "$tool" >/dev/null 2>&1; then
  tool="$root/dist/appimagetool"
  [ -x "$tool" ] || { curl -fL -o "$tool" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage; chmod +x "$tool"; }
fi
# CI runners have no FUSE, so the tool unpacks itself instead of mounting.
ARCH=x86_64 "$tool" --appimage-extract-and-run "$app" "dist/Scoobert-$version-x86_64.AppImage"
echo "Built dist/$tarname.tar.gz and dist/Scoobert-$version-x86_64.AppImage"
