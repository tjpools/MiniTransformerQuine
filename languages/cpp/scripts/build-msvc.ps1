$ErrorActionPreference = "Stop"

$src = "src"
$include = "include"
$outdir = "build/msvc"

if (!(Test-Path $outdir)) {
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
}

$files = Get-ChildItem $src -Filter *.cpp | ForEach-Object { $_.FullName }

$cl = @(
    "/EHsc",
    "/std:c++20",
    "/I$include",
    "/Fe$outdir/mini-transformer-quine.exe"
)

Write-Host "Compiling mini-transformer-quine..."
cl $cl $files
