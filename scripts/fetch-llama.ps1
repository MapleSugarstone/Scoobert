# Puts the llama.cpp server into vendor/llama so a release can bundle it.
# By default it copies the build that `winget install ggml.llamacpp` installed. With -Release <build number> it
# downloads that CPU build from the llama.cpp GitHub releases instead, which is what the release workflow does,
# and adds the Vulkan backend from the same release for the graphics card setting. llama-server loads a backend
# only when its library is present, and Scoobert keeps the card off unless the setting is on.
param([string]$Release = '')
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot '..'
$out = Join-Path $root 'vendor\llama'
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null

if ($Release) {
	$zip = Join-Path ([IO.Path]::GetTempPath()) "llama-$Release-win.zip"
	$url = "https://github.com/ggml-org/llama.cpp/releases/download/$Release/llama-$Release-bin-win-cpu-x64.zip"
	Invoke-WebRequest -UseBasicParsing $url -OutFile $zip
	$src = Join-Path ([IO.Path]::GetTempPath()) "llama-$Release-win"
	Remove-Item -Recurse -Force $src -ErrorAction SilentlyContinue
	Expand-Archive $zip -DestinationPath $src
	$source = Get-ChildItem $src -Recurse -Filter 'llama-server.exe' | Select-Object -First 1 | ForEach-Object DirectoryName
	$vulkanZip = Join-Path ([IO.Path]::GetTempPath()) "llama-$Release-win-vulkan.zip"
	$vulkanUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$Release/llama-$Release-bin-win-vulkan-x64.zip"
	Invoke-WebRequest -UseBasicParsing $vulkanUrl -OutFile $vulkanZip
	$vulkanSrc = Join-Path ([IO.Path]::GetTempPath()) "llama-$Release-win-vulkan"
	Remove-Item -Recurse -Force $vulkanSrc -ErrorAction SilentlyContinue
	Expand-Archive $vulkanZip -DestinationPath $vulkanSrc
	Get-ChildItem $vulkanSrc -Recurse -Filter 'ggml-vulkan.dll' | Select-Object -First 1 | Copy-Item -Destination $source
	$origin = "Downloaded from $url, with ggml-vulkan.dll from $vulkanUrl"
} else {
	$pkg = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') -Directory -Filter 'ggml.llamacpp*' -ErrorAction SilentlyContinue | Select-Object -First 1
	if (-not $pkg) { throw 'llama.cpp is not installed. Run: winget install ggml.llamacpp, or pass -Release b<number>.' }
	$source = $pkg.FullName
	$origin = "Copied from $($pkg.Name)"
}

$keep = Get-ChildItem $source -File | Where-Object {
	$_.Name -eq 'llama-server.exe' -or
	$_.Name -like 'LICENSE*' -or
	($_.Name -like '*.dll' -and $_.Name -notlike 'ggml-cuda*' -and ($_.Name -notlike 'llama-*-impl.dll' -or $_.Name -eq 'llama-server-impl.dll'))
}
$keep | Copy-Item -Destination $out
if (-not (Test-Path (Join-Path $out 'LICENSE*'))) {
	Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE' -OutFile (Join-Path $out 'LICENSE-llama.cpp')
}
# Records exactly which build was bundled, so a release can be traced back and compared.
# cmd merges the server's stderr, which PowerShell would otherwise treat as a terminating error.
$version = (cmd /c "`"$(Join-Path $out 'llama-server.exe')`" --version 2>&1" | Select-String 'version:').Line.Trim()
$hashes = Get-ChildItem $out -File | Where-Object { $_.Extension -in '.exe', '.dll' } | ForEach-Object { '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name }
Set-Content -Path (Join-Path $out 'VERSION.txt') -Encoding ascii -Value (@("llama.cpp $version", $origin, '') + $hashes)
$size = ($keep | Measure-Object Length -Sum).Sum / 1MB
Write-Output ("Put {0} files ({1:N0} MB) in vendor/llama. {2}" -f $keep.Count, $size, $origin)
