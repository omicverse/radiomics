# Optional helper for this repository's locally isolated toolchain and virtualenv.
# Dot-source this file: . .\tools\windows-rust-env.ps1
$radiomicsRoot = Split-Path -Parent $PSScriptRoot
$radiomicsBuildTools = Join-Path $radiomicsRoot '.build-tools'
if (Test-Path -LiteralPath (Join-Path $radiomicsBuildTools 'cargo\bin\cargo.exe')) {
    $env:CARGO_HOME = Join-Path $radiomicsBuildTools 'cargo'
    $env:RUSTUP_HOME = Join-Path $radiomicsBuildTools 'rustup'
    $env:PATH = (Join-Path $env:CARGO_HOME 'bin') + [IO.Path]::PathSeparator + $env:PATH
}
if (Test-Path -LiteralPath (Join-Path $radiomicsRoot '.venv\Scripts\python.exe')) {
    $env:VIRTUAL_ENV = Join-Path $radiomicsRoot '.venv'
    $env:PATH = (Join-Path $env:VIRTUAL_ENV 'Scripts') + [IO.Path]::PathSeparator + $env:PATH
}
