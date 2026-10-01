# Builds the release binary and the Windows installer in dist/.
# Run scripts/fetch-llama.ps1 first so vendor/llama holds the server.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $root
try {
	cargo build --release
	if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
	if (-not (Test-Path 'vendor\llama\llama-server.exe')) { throw 'vendor\llama is empty. Run scripts\fetch-llama.ps1 first.' }
	$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value

	$stage = Join-Path $root 'dist\Scoobert'
	Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
	New-Item -ItemType Directory -Force (Join-Path $stage 'llama') | Out-Null
	Copy-Item 'target\release\scoobert.exe' $stage
	Copy-Item 'LICENSE' $stage
	Copy-Item 'vendor\llama\*' (Join-Path $stage 'llama')

	$nsis = Get-Command makensis -ErrorAction SilentlyContinue | ForEach-Object Source
	if (-not $nsis) {
		$nsis = Get-ChildItem "$env:ProgramFiles*\NSIS", "$env:LOCALAPPDATA\electron-builder\Cache" -Recurse -Filter makensis.exe -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object FullName
	}
	if (-not $nsis) { throw 'NSIS is not installed. Run: winget install NSIS.NSIS' }
	# The installer's translations are UTF-8 without a byte order mark, which makensis would otherwise read as ANSI.
	& $nsis '/INPUTCHARSET' 'UTF8' "/DVERSION=$version" "/DSOURCE=$stage" 'packaging\windows\scoobert.nsi'
	if ($LASTEXITCODE -ne 0) { throw 'makensis failed' }
	$setup = Get-Item "dist\Scoobert-Setup-$version.exe"
	Write-Output ("Built {0} ({1:N0} MB)" -f $setup.FullName, ($setup.Length / 1MB))

	# The portable copy is the same files plus portable.txt, which keeps all data inside its folder.
	$portable = Join-Path $root 'dist\portable\Scoobert'
	Remove-Item -Recurse -Force (Join-Path $root 'dist\portable') -ErrorAction SilentlyContinue
	New-Item -ItemType Directory -Force $portable | Out-Null
	Copy-Item "$stage\*" $portable -Recurse
	Copy-Item 'packaging\portable.txt' $portable
	$zip = Join-Path $root "dist\Scoobert-$version-portable-windows.zip"
	Remove-Item $zip -ErrorAction SilentlyContinue
	Compress-Archive -Path $portable -DestinationPath $zip
	Write-Output ("Built {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
} finally {
	Pop-Location
}
