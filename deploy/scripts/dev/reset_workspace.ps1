[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$KeepRawJobs
)

$ErrorActionPreference = "Stop"

function Resolve-InWorkspacePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $target = Join-Path $BaseDir $RelativePath
    $resolved = [System.IO.Path]::GetFullPath($target)
    $baseResolved = [System.IO.Path]::GetFullPath($BaseDir)

    if (-not $resolved.StartsWith($baseResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe path detected outside workspace: $resolved"
    }

    return $resolved
}

function Clear-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToClear
    )

    if (-not (Test-Path -LiteralPath $PathToClear)) {
        return
    }

    Get-ChildItem -LiteralPath $PathToClear -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($PSCmdlet.ShouldProcess($_.FullName, "Remove")) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
}

# deploy/scripts/dev -> project root is ../../..
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
Write-Host "[INFO] Project root: $projectRoot"

$targets = @(
    "experiments\artifacts",
    "data\processed",
    "hf_cache"
)

if (-not $KeepRawJobs) {
    $targets += "data\raw\jobs"
}

foreach ($relative in $targets) {
    $abs = Resolve-InWorkspacePath -BaseDir $projectRoot -RelativePath $relative
    if (-not (Test-Path -LiteralPath $abs)) {
        New-Item -ItemType Directory -Path $abs -Force | Out-Null
        Write-Host "[INFO] Created missing directory: $relative"
        continue
    }

    Write-Host "[INFO] Clearing: $relative"
    Clear-DirectoryContents -PathToClear $abs
}

# Ensure base output folders exist after reset
$ensureDirs = @(
    "experiments\artifacts\matching",
    "experiments\artifacts\evaluation",
    "experiments\artifacts\chatbot",
    "experiments\artifacts\models",
    "data\processed",
    "data\raw\jobs"
)

foreach ($relative in $ensureDirs) {
    $abs = Resolve-InWorkspacePath -BaseDir $projectRoot -RelativePath $relative
    if (-not (Test-Path -LiteralPath $abs)) {
        New-Item -ItemType Directory -Path $abs -Force | Out-Null
    }
}

Write-Host "[DONE] Workspace outputs reset completed."
Write-Host "[NOTE] Kept folders: data/raw/cv_samples, data/reference, source code."
if ($KeepRawJobs) {
    Write-Host "[NOTE] Raw job files were preserved (--KeepRawJobs)."
}
