param(
    [ValidateSet('prepare','inventory','analyze','demo','verify-package','test')]
    [string]$Action = 'verify-package',
    [string]$DataRoot,
    [string]$Manifest,
    [string]$SessionRecord,
    [string]$OutputDirectory,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExecutable) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $PythonExecutable = $bundledPython }
    else { $PythonExecutable = (Get-Command python -ErrorAction Stop).Source }
}
if ($Action -eq 'test') {
    & $PythonExecutable -X utf8 -m unittest discover -s $PSScriptRoot -p 'test_ei2.py'
    exit $LASTEXITCODE
}
$runnerArguments = @('-X','utf8',(Join-Path $PSScriptRoot 'ei2.py'),$Action)
if ($Action -in @('inventory','analyze')) {
    if (-not $DataRoot) { throw 'Supply -DataRoot pointing to the copied EI2 session folder.' }
    $runnerArguments += @('--data-root',$DataRoot)
}
if ($Action -eq 'analyze') {
    if (-not $Manifest -or -not $SessionRecord) { throw 'Supply -Manifest and -SessionRecord transcribed from the completed input sheet.' }
    $runnerArguments += @('--manifest',$Manifest,'--session',$SessionRecord)
}
if ($Action -in @('inventory','analyze','demo','prepare')) {
    if (-not $OutputDirectory) { throw 'Supply a new -OutputDirectory. Existing folders are never overwritten.' }
    $runnerArguments += @('--out',$OutputDirectory)
}
& $PythonExecutable @runnerArguments
exit $LASTEXITCODE
