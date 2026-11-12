param(
    [ValidateSet('Configure', 'Check', 'Run')][string]$Action = 'Check',
    [ValidateSet('All', 'Databento', 'OANDA')][string]$Provider = 'All',
    [ValidateRange(1024, 65535)][int]$Port = 8877,
    [string]$CredentialsPath
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'This credential helper requires Windows.' }
# A parent PowerShell 7 process may pass module paths incompatible with 5.1.
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -Global -ErrorAction Stop
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $CredentialsPath) {
    $CredentialsPath = Join-Path $env:LOCALAPPDATA 'CrossVenue\provider-access.clixml'
}
$names = @('DATABENTO_API_KEY', 'OANDA_ACCOUNT_ID', 'OANDA_ACCESS_TOKEN')

function Read-ProviderStore {
    if (-not (Test-Path -LiteralPath $CredentialsPath)) {
        return [pscustomobject]@{ SchemaVersion = 1; Entries = @{}; OandaEnvironment = 'practice' }
    }
    try { $saved = Import-Clixml -LiteralPath $CredentialsPath }
    catch { throw 'Cannot decrypt the local provider credentials. Use Data Setup with the original Windows account, or configure them again.' }
    if ($saved.SchemaVersion -ne 1 -or $saved.Entries -isnot [System.Collections.IDictionary] -or
        $saved.OandaEnvironment -notin @('practice', 'live')) { throw 'Invalid provider credential store.' }
    foreach ($name in $saved.Entries.Keys) {
        if ($name -notin $names -or $saved.Entries[$name] -isnot [Security.SecureString]) {
            throw 'Invalid provider credential entry.'
        }
    }
    return $saved
}

function Get-PlainSecret([Security.SecureString]$Secret) {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
}

$saved = Read-ProviderStore
if ($Action -eq 'Configure') {
    Write-Output 'CrossVenue - Market Data Setup'
    Write-Output 'Create your provider accounts first. Enter credentials here, never in chat or source files.'
    Write-Output 'Press Enter on an empty credential prompt to keep the existing value.'
    if ($Provider -in @('All', 'Databento')) {
        $key = Read-Host 'Databento API key (stocks and futures)' -AsSecureString
        if ($key.Length -gt 0) { $saved.Entries['DATABENTO_API_KEY'] = $key }
    }
    if ($Provider -in @('All', 'OANDA')) {
        $account = Read-Host 'OANDA v20 account ID' -AsSecureString
        $token = Read-Host 'OANDA personal access token' -AsSecureString
        if ($account.Length -gt 0) { $saved.Entries['OANDA_ACCOUNT_ID'] = $account }
        if ($token.Length -gt 0) { $saved.Entries['OANDA_ACCESS_TOKEN'] = $token }
        $choice = Read-Host ('OANDA environment: practice or live [Enter keeps ' + $saved.OandaEnvironment + ']')
        if ($choice) {
            $choice = $choice.Trim().ToLowerInvariant()
            if ($choice -notin @('practice', 'live')) { throw 'Environment must be practice or live. Nothing was saved.' }
            $saved.OandaEnvironment = $choice
        }
    }
    if ($saved.Entries.Count -eq 0) {
        Write-Output 'No credentials entered. No credential file or configuration was changed.'
        return
    }
    $directory = Split-Path -Parent ([IO.Path]::GetFullPath($CredentialsPath))
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    # On Windows Export-Clixml encrypts SecureStrings with current-user DPAPI.
    $temporary = Join-Path $directory ('provider-access-' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $saved | Export-Clixml -LiteralPath $temporary -Depth 5
        $security = New-Object Security.AccessControl.FileSecurity
        foreach ($sid in @([Security.Principal.WindowsIdentity]::GetCurrent().User,
                          [Security.Principal.SecurityIdentifier]::new('S-1-5-18'))) {
            $security.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
                $sid, [Security.AccessControl.FileSystemRights]::FullControl,
                [Security.AccessControl.AccessControlType]::Allow))
        }
        $security.SetAccessRuleProtection($true, $false)
        Set-Acl -LiteralPath $temporary -AclObject $security
        [IO.File]::Copy($temporary, $CredentialsPath, $true)
        Set-Acl -LiteralPath $CredentialsPath -AclObject $security
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
    $configPath = Join-Path $projectRoot 'config\cross_asset_paper.json'
    if ($Provider -in @('All', 'OANDA') -and $saved.Entries.Contains('OANDA_ACCOUNT_ID') -and
        $saved.Entries.Contains('OANDA_ACCESS_TOKEN') -and (Test-Path -LiteralPath $configPath)) {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        $changed = $false
        foreach ($feed in $config.providers) {
            if ($feed.type -eq 'oanda' -and $feed.environment -ne $saved.OandaEnvironment) {
                $feed.environment = $saved.OandaEnvironment
                $changed = $true
            }
        }
        if ($changed) {
            $backupRoot = Join-Path $projectRoot 'data\backups'
            New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
            Copy-Item -LiteralPath $configPath -Destination (Join-Path $backupRoot ('provider-config-' + [guid]::NewGuid().ToString('N') + '.json'))
            [IO.File]::WriteAllText($configPath, ($config | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
        }
    }
    Write-Output 'Credentials saved encrypted outside the project for this Windows account.'
    Write-Output 'After safely stopping the current paper backend, reopen Paper Trading to load them.'
    Write-Output 'Provider licenses and data entitlements are separate; this wizard does not purchase or validate them.'
    return
}

if ($Action -eq 'Check') {
    foreach ($name in $names) {
        $present = [bool][Environment]::GetEnvironmentVariable($name, 'Process') -or
                   [bool][Environment]::GetEnvironmentVariable($name, 'User') -or
                   ($saved.Entries.Contains($name) -and $saved.Entries[$name].Length -gt 0)
        Write-Output ($name + ': ' + $(if ($present) { 'configured (value hidden)' } else { 'missing' }))
    }
    Write-Output 'Offline credential-presence check only; live entitlements have not been verified.'
    return
}

$python = Join-Path $projectRoot '.venv-paper\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Dedicated Python environment missing. Run the installer first.' }
$entrypoint = Join-Path $projectRoot 'launch.py'
if (-not (Test-Path -LiteralPath $entrypoint)) { $entrypoint = Join-Path $projectRoot 'launcher_with_ui.py' }
$configPath = Join-Path $projectRoot 'config\cross_asset_paper.json'
if (-not (Test-Path -LiteralPath $entrypoint) -or -not (Test-Path -LiteralPath $configPath)) {
    throw 'Paper launcher or reviewed private configuration is missing.'
}
$previous = @{}
try {
    foreach ($name in $names) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        if (-not $previous[$name]) {
            $value = if ($saved.Entries.Contains($name)) { Get-PlainSecret $saved.Entries[$name] }
                     else { [Environment]::GetEnvironmentVariable($name, 'User') }
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            $value = $null
        }
    }
    Push-Location -LiteralPath $projectRoot
    try { & $python -B $entrypoint --config $configPath --port $Port; $exitCode = $LASTEXITCODE }
    finally { Pop-Location }
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
}
exit $exitCode
