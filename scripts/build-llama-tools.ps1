# Builds llama.cpp's control vector tool, which the model lab's steering uses and llama.cpp's releases leave out,
# and puts it in vendor/llama beside the server. It builds from the same release as the bundled server, with the
# patches in scripts/llama-patches, as one static program for the processor so it needs no other llama.cpp files.
# Usage: scripts/build-llama-tools.ps1 -Release b11193
param([Parameter(Mandatory)][string]$Release)
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot '..'
$out = Join-Path $root 'vendor\llama'
if (-not (Test-Path (Join-Path $out 'llama-server.exe'))) { throw 'vendor\llama is empty. Run scripts\fetch-llama.ps1 first.' }

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
	$cmake = Get-ChildItem "$env:ProgramFiles\Microsoft Visual Studio" -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cmake) { throw 'CMake was not found. Install Visual Studio with C++ and CMake.' }

# MSBuild refuses to build inside the temporary folder, so the work happens under target.
$work = Join-Path $root "target\llama-tools-$Release"
$src = Join-Path $work 'src'
if (-not (Test-Path (Join-Path $src '.git'))) {
	Remove-Item -Recurse -Force $src -ErrorAction SilentlyContinue
	git -c core.longpaths=true clone --quiet --depth 1 --branch $Release https://github.com/ggml-org/llama.cpp $src
	if ($LASTEXITCODE -ne 0) { throw "Could not download llama.cpp $Release" }
}
git -C $src checkout --quiet -- .
foreach ($patch in Get-ChildItem (Join-Path $PSScriptRoot 'llama-patches') -Filter *.patch) {
	git -C $src apply --whitespace=nowarn $patch.FullName
	if ($LASTEXITCODE -ne 0) { throw "Could not apply $($patch.Name)" }
}

$build = Join-Path $work 'build'
& $cmake -S $src -B $build -A x64 -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON `
	-DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF `
	-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'CMake could not configure llama.cpp' }
& $cmake --build $build --config Release --target llama-cvector-generator -j 4 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The control vector tool did not build' }
$exe = Get-ChildItem $build -Recurse -Filter 'llama-cvector-generator.exe' | Select-Object -First 1
Copy-Item $exe.FullName $out
Add-Content -Path (Join-Path $out 'VERSION.txt') -Encoding ascii -Value ("{0}  llama-cvector-generator.exe (built from {1} with scripts/llama-patches)" -f (Get-FileHash $exe.FullName -Algorithm SHA256).Hash.ToLower(), $Release)
Write-Output ("Built llama-cvector-generator.exe ({0:N1} MB) into vendor/llama" -f ($exe.Length / 1MB))
