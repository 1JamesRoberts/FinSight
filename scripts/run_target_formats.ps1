param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$DataDir,
    [Parameter(Mandatory=$true)][string]$JobDir
)
# Run from Task Scheduler so neither SSH nor the Mac owns the training process.
$ErrorActionPreference = 'Stop'
$python = Join-Path (Split-Path $DataDir) '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
$state = [ordered]@{
    status='running'; stage='starting'; pid=$PID;
    started_utc=(Get-Date).ToUniversalTime().ToString('o');
    source=$SourceRoot; data=$DataDir
}
function Write-State([string]$stage) {
    $state.stage = $stage
    $state.updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    [IO.File]::WriteAllText((Join-Path $JobDir 'state.tmp'), ($state | ConvertTo-Json -Depth 6))
    Move-Item -Force (Join-Path $JobDir 'state.tmp') (Join-Path $JobDir 'state.json')
}
function Invoke-Stage([string]$name, [string[]]$Arguments) {
    Write-State $name
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $SourceRoot -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $JobDir ($name + '.stdout.log')) `
        -RedirectStandardError (Join-Path $JobDir ($name + '.stderr.log'))
    if ($process.ExitCode -ne 0) { throw "$name failed with exit $($process.ExitCode); inspect its logs" }
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class FinSightTrainingAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
# Hold a sleep-prevention request only while this job runs; do not change power settings.
$awake = [FinSightTrainingAwake]::SetThreadExecutionState([uint32]2147483649)
try {
    if ($awake -eq 0) { throw 'Could not hold the job sleep-prevention request' }
    Set-Location $SourceRoot
    $env:PYTHONPATH = Join-Path $SourceRoot 'src'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $revision = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve execution snapshot revision' }
    if (git status --porcelain) { throw 'Execution snapshot is not clean' }
    $state.revision = $revision
    $plan = [IO.File]::ReadAllText((Join-Path $SourceRoot 'configs\target-format-plan.json')) | ConvertFrom-Json
    $cache = Join-Path $DataDir ('interim\transformer-checkpoint-cache\models--' + $plan.asset.model_id.Replace('/','--') + '\snapshots\' + $plan.asset.revision)
    Invoke-Stage 'ruff-check' @('-m','ruff','check','.')
    Invoke-Stage 'ruff-format' @('-m','ruff','format','--check','.')
    Invoke-Stage 'pytest' @('-m','pytest','-q')
    Invoke-Stage 'preparation-check' @('-m','finsight.sentiment.experiments.target_format_preparation','--data-dir',$DataDir,'--tokenizer-dir',$cache,'--check')
    Invoke-Stage 'training' @('-u','-m','finsight.sentiment.experiments.target_format_training','--data-dir',$DataDir,'--tokenizer-dir',$cache,'--state',(Join-Path $JobDir 'trials.json'))
    $exports = Join-Path $JobDir 'exports'
    New-Item -ItemType Directory -Force -Path $exports | Out-Null
    Copy-Item (Join-Path $SourceRoot 'reports\experiments\target-format.json') $exports
    Copy-Item (Join-Path $SourceRoot 'reports\experiments\target-format.md') $exports
    Copy-Item (Join-Path $SourceRoot 'reports\experiments\target-format-runs') $exports -Recurse
    $state.status = 'complete'
    $state.exports = $exports
    Write-State 'complete'
} catch {
    $state.status = 'failed'
    $state.error = $_.Exception.Message
    Write-State 'failed'
    throw
} finally {
    [FinSightTrainingAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
