$ErrorActionPreference = "Stop"

$port = if ($env:API_PORT) { [int]$env:API_PORT } else { 8010 }
$maxWaitSeconds = 15

function Get-PortOwners {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $listeners) {
        return @()
    }
    return @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
}

# Clean up any stale process that is still holding the API port.
$owners = Get-PortOwners
if ($owners.Count -gt 0) {
    $owners | ForEach-Object {
        if (-not $_) {
            return
        }
        if (-not (Get-Process -Id $_ -ErrorAction SilentlyContinue)) {
            return
        }
        try {
            Stop-Process -Id $_ -Force -ErrorAction Stop
        } catch {
            Write-Warning "Could not stop process $_ on port ${port}: $($_.Exception.Message)"
        }
    }
}

$deadline = (Get-Date).AddSeconds($maxWaitSeconds)
while ((Get-Date) -lt $deadline) {
    $owners = Get-PortOwners
    if ($owners.Count -eq 0) {
        break
    }
    foreach ($owner in $owners) {
        if (-not $owner) {
            continue
        }
        if (-not (Get-Process -Id $owner -ErrorAction SilentlyContinue)) {
            continue
        }
        try {
            Stop-Process -Id $owner -Force -ErrorAction Stop
        } catch {
            try {
                Write-Warning "Retry stop failed for process $owner on port ${port}: $($_.Exception.Message)"
            } catch {}
        }
    }
    Start-Sleep -Milliseconds 750
}

$owners = Get-PortOwners
if ($owners.Count -gt 0) {
    throw "Port $port is still in use by process id(s): $($owners -join ', ')."
}

python -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port $port
