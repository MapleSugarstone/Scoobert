#!/usr/bin/env bash
# Builds llama.cpp's control vector tool, which the model lab's steering uses and llama.cpp's releases leave out,
# and puts it in vendor/llama beside the server. It builds from the same release as the bundled server, with the
# patches in scripts/llama-patches, as one static program for the processor so it needs no other llama.cpp files.
# Usage: scripts/build-llama-tools.sh b11193
set -euo pipefail
release="${1:?Pass a llama.cpp release, for example b11193}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/vendor/llama"
[ -x "$out/llama-server" ] || { echo "vendor/llama is empty. Run scripts/fetch-llama.sh first." >&2; exit 1; }
work="$root/target/llama-tools-$release"
src="$work/src"
if [ ! -d "$src/.git" ]; then
  rm -rf "$src"
  git clone --quiet --depth 1 --branch "$release" https://github.com/ggml-org/llama.cpp "$src"
fi
git -C "$src" checkout --quiet -- .
for patch in "$root"/scripts/llama-patches/*.patch; do
  git -C "$src" apply --whitespace=nowarn "$patch"
done
cmake -S "$src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > /dev/null
cmake --build "$work/build" --target llama-cvector-generator -j "$(nproc)" > /dev/null
exe="$(find "$work/build" -name llama-cvector-generator -type f | head -n 1)"
cp "$exe" "$out/"
chmod +x "$out/llama-cvector-generator"
echo "$(sha256sum "$out/llama-cvector-generator" | cut -d' ' -f1)  llama-cvector-generator (built from $release with scripts/llama-patches)" >> "$out/VERSION.txt"
echo "Built llama-cvector-generator ($(du -h "$out/llama-cvector-generator" | cut -f1)) into vendor/llama"
