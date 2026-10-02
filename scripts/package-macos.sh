#!/usr/bin/env bash
# Builds the release binary, then Scoobert.app and a disk image in dist/, for Apple Silicon Macs.
# Run scripts/fetch-llama.sh first so vendor/llama holds the server.
# The app is signed ad hoc, which Apple Silicon needs to run it at all. It is not notarized, so on first open macOS
# asks the user to approve it in System Settings, Privacy & Security.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --release
[ -x vendor/llama/llama-server ] || { echo "vendor/llama is empty. Run scripts/fetch-llama.sh first." >&2; exit 1; }
version="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
# The app asks for the macOS version the bundled server was built for, so it never opens where the server cannot run.
minos="$(otool -l vendor/llama/llama-server | awk '/LC_BUILD_VERSION/ { found = 1 } found && $1 == "minos" { print $2; exit }')"
minos="${minos:-13.0}"
mkdir -p dist

app=dist/Scoobert.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS/llama" "$app/Contents/Resources/llama"
cp target/release/scoobert "$app/Contents/MacOS/"
# Scoobert looks for the server in a llama folder beside its own program. Code signing allows only programs and
# libraries there, so the licenses and version list go to Resources.
for f in vendor/llama/*; do
  case "$f" in
    *.dylib | *.metallib) cp -P "$f" "$app/Contents/MacOS/llama/" ;;
    *) if file -b "$f" | grep -q Mach-O; then cp "$f" "$app/Contents/MacOS/llama/"; else cp "$f" "$app/Contents/Resources/llama/"; fi ;;
  esac
done
cp packaging/macos/Scoobert.icns LICENSE "$app/Contents/Resources/"
sed -e "s/@VERSION@/$version/g" -e "s/@MINOS@/$minos/g" packaging/macos/Info.plist > "$app/Contents/Info.plist"
chmod +x "$app/Contents/MacOS/scoobert" "$app/Contents/MacOS/llama/"llama-*

# Each program and library is signed before the app around them.
find "$app/Contents/MacOS/llama" -type f \( -perm -u+x -o -name '*.dylib' \) -exec codesign --force --sign - {} \;
codesign --force --sign - "$app/Contents/MacOS/scoobert"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"

# The bundled server has to start from inside the app, which proves its libraries are found there.
"$app/Contents/MacOS/llama/llama-server" --version

dmg="dist/Scoobert-$version-macos-arm64.dmg"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"
rm -f "$dmg"
hdiutil create -volname "Scoobert $version" -srcfolder "$stage" -fs HFS+ -format UDZO -ov "$dmg" > /dev/null
echo "Built $dmg ($(du -h "$dmg" | cut -f1)) for macOS $minos and later"
