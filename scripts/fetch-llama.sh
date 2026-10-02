#!/usr/bin/env bash
# Downloads a llama.cpp build into vendor/llama so a release can bundle it. On Linux it is the CPU build with the
# Vulkan backend from the same release for the graphics card setting, which loads only when the system has a Vulkan
# driver. On macOS it is the Apple Silicon build, which runs on the graphics chip through Metal.
# Usage: scripts/fetch-llama.sh b11193
set -euo pipefail
release="${1:?Pass a llama.cpp release, for example b11193}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/vendor/llama"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [ "$(uname -s)" = Darwin ]; then
  mac=1
  url="https://github.com/ggml-org/llama.cpp/releases/download/$release/llama-$release-bin-macos-arm64.tar.gz"
  sha() { shasum -a 256 "$@"; }
else
  mac=0
  url="https://github.com/ggml-org/llama.cpp/releases/download/$release/llama-$release-bin-ubuntu-x64.tar.gz"
  sha() { sha256sum "$@"; }
fi

curl -fL --retry 3 -o "$tmp/llama.tar.gz" "$url"
mkdir -p "$tmp/src"
tar -xzf "$tmp/llama.tar.gz" -C "$tmp/src"
bin="$(dirname "$(find "$tmp/src" -name llama-server -type f | head -n 1)")"

rm -rf "$out"
mkdir -p "$out"
cp "$bin/llama-server" "$out/"
# The model lab converts models with llama-quantize.
cp "$bin/llama-quantize" "$out/"
if [ "$mac" = 1 ]; then
  # The server loads the ggml backends and libllama from its own folder.
  find "$bin" -maxdepth 1 \( -name '*.dylib' -o -name '*.metallib' \) -exec cp -P {} "$out/" \;
else
  find "$bin" -maxdepth 1 -name '*.so*' ! -name '*vulkan*' ! -name '*cuda*' -exec cp -P {} "$out/" \;
  vulkan_url="https://github.com/ggml-org/llama.cpp/releases/download/$release/llama-$release-bin-ubuntu-vulkan-x64.tar.gz"
  curl -fL --retry 3 -o "$tmp/vulkan.tar.gz" "$vulkan_url"
  mkdir -p "$tmp/vulkan"
  tar -xzf "$tmp/vulkan.tar.gz" -C "$tmp/vulkan"
  find "$tmp/vulkan" -name 'libggml-vulkan.so*' -exec cp -P {} "$out/" \;
fi
find "$tmp/src" -maxdepth 3 -name 'LICENSE*' -exec cp {} "$out/" \; || true
[ -e "$out/LICENSE" ] || curl -fsSL -o "$out/LICENSE-llama.cpp" https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE
chmod +x "$out/llama-server" "$out/llama-quantize"

{
  echo "llama.cpp $release"
  echo "Downloaded from $url"
  echo
  (cd "$out" && sha llama-server llama-quantize ./*.so* ./*.dylib 2>/dev/null || true)
} > "$out/VERSION.txt"
echo "Put $(ls "$out" | wc -l | tr -d ' ') files ($(du -sh "$out" | cut -f1)) in vendor/llama from $url"
