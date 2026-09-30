#!/usr/bin/env bash
# Downloads a llama.cpp CPU build for Linux into vendor/llama so a release can bundle it.
# Usage: scripts/fetch-llama.sh b11193
set -euo pipefail
release="${1:?Pass a llama.cpp release, for example b11193}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/vendor/llama"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

url="https://github.com/ggml-org/llama.cpp/releases/download/$release/llama-$release-bin-ubuntu-x64.zip"
curl -fL --retry 3 -o "$tmp/llama.zip" "$url"
unzip -q "$tmp/llama.zip" -d "$tmp/src"
bin="$(dirname "$(find "$tmp/src" -name llama-server -type f | head -n 1)")"

rm -rf "$out"
mkdir -p "$out"
cp "$bin/llama-server" "$out/"
# The server loads the ggml backends and libllama from its own folder.
find "$bin" -maxdepth 1 -name '*.so*' ! -name '*vulkan*' ! -name '*cuda*' -exec cp -P {} "$out/" \;
find "$tmp/src" -maxdepth 3 -name 'LICENSE*' -exec cp {} "$out/" \; || true
[ -e "$out/LICENSE" ] || curl -fsSL -o "$out/LICENSE-llama.cpp" https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE
chmod +x "$out/llama-server"

{
  echo "llama.cpp $release"
  echo "Downloaded from $url"
  echo
  (cd "$out" && sha256sum llama-server ./*.so* 2>/dev/null || true)
} > "$out/VERSION.txt"
echo "Put $(ls "$out" | wc -l) files ($(du -sh "$out" | cut -f1)) in vendor/llama from $url"
