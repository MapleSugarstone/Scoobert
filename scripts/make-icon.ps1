# Builds the app icon and in-app logo from assets/icon-source.png with nearest-neighbor scaling.
# Outputs: assets/icon.png (256 px), assets/icon.ico (16 to 256 px), assets/logo.png (cropped, 1x).
Add-Type -AssemblyName System.Drawing
$root = Join-Path $PSScriptRoot '..'
$src = [System.Drawing.Bitmap]::FromFile((Resolve-Path (Join-Path $root 'assets\icon-source.png')))

# Crop to the opaque pixels so the drawing fills each icon size.
$minX = $src.Width; $minY = $src.Height; $maxX = -1; $maxY = -1
for ($y = 0; $y -lt $src.Height; $y++) {
	for ($x = 0; $x -lt $src.Width; $x++) {
		if ($src.GetPixel($x, $y).A -gt 0) {
			if ($x -lt $minX) { $minX = $x }; if ($x -gt $maxX) { $maxX = $x }
			if ($y -lt $minY) { $minY = $y }; if ($y -gt $maxY) { $maxY = $y }
		}
	}
}
$cw = $maxX - $minX + 1
$ch = $maxY - $minY + 1
$side = [Math]::Max($cw, $ch)

# Whole-number scale factors keep every source pixel square; sizes below the drawing sample every nth pixel.
function Render([int]$size, [int]$width = $size, [int]$height = $size) {
	$scale = $size / $side
	if ($scale -ge 1) { $scale = [Math]::Floor($scale) }
	$w = [int][Math]::Floor($cw * $scale)
	$h = [int][Math]::Floor($ch * $scale)
	$ox = [int][Math]::Floor(($width - $w) / 2)
	$oy = [int][Math]::Floor(($height - $h) / 2)
	$bmp = New-Object System.Drawing.Bitmap $width, $height, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
	for ($ty = 0; $ty -lt $h; $ty++) {
		$sy = $minY + [Math]::Min($ch - 1, [int][Math]::Floor($ty / $scale))
		for ($tx = 0; $tx -lt $w; $tx++) {
			$sx = $minX + [Math]::Min($cw - 1, [int][Math]::Floor($tx / $scale))
			$bmp.SetPixel($ox + $tx, $oy + $ty, $src.GetPixel($sx, $sy))
		}
	}
	return $bmp
}

function PngBytes($bmp) {
	$ms = New-Object System.IO.MemoryStream
	$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
	return , $ms.ToArray()
}

$sizes = 16, 24, 32, 48, 64, 128, 256
$images = @{}
foreach ($s in $sizes) { $images[$s] = PngBytes (Render $s) }
[IO.File]::WriteAllBytes((Join-Path $root 'assets\icon.png'), $images[256])

# An ICO file is a directory of PNG images, which Windows Vista and later accept.
$ms = New-Object System.IO.MemoryStream
$bw = New-Object System.IO.BinaryWriter $ms
$bw.Write([UInt16]0); $bw.Write([UInt16]1); $bw.Write([UInt16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
foreach ($s in $sizes) {
	$dim = if ($s -ge 256) { 0 } else { $s }
	$bw.Write([Byte]$dim); $bw.Write([Byte]$dim); $bw.Write([Byte]0); $bw.Write([Byte]0)
	$bw.Write([UInt16]1); $bw.Write([UInt16]32)
	$bw.Write([UInt32]$images[$s].Length); $bw.Write([UInt32]$offset)
	$offset += $images[$s].Length
}
foreach ($s in $sizes) { $bw.Write($images[$s]) }
$bw.Flush()
[IO.File]::WriteAllBytes((Join-Path $root 'assets\icon.ico'), $ms.ToArray())

# The in-app logo is the cropped drawing at 1x; the UI scales it by whole numbers.
$logo = Render $side $cw $ch
$logo.Save((Join-Path $root 'assets\logo.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$src.Dispose()
Write-Output "Cropped drawing: ${cw}x${ch}. Wrote assets/icon.png, assets/icon.ico ($($sizes -join ', ') px), assets/logo.png."
