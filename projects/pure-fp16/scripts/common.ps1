$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}

$script:ScriptDir = Split-Path -Parent $PSCommandPath

function Get-ExperimentRoot {
  return (Resolve-Path (Join-Path $script:ScriptDir "..")).Path
}

function Get-RepoRoot {
  $experimentRoot = Get-ExperimentRoot
  return (Resolve-Path (Join-Path $experimentRoot "..\..")).Path
}

function Get-AdbSerialArgs {
  param([string] $Serial = "")
  if ([string]::IsNullOrWhiteSpace($Serial)) {
    return @()
  }
  return @("-s", $Serial)
}

function Assert-LpbqR4PerformancePathCancelled {
  param([string[]] $RequestedKnobs = @())

  $active = @($RequestedKnobs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($active.Count -gt 0) {
    throw "R4/FWHT path is cancelled for the LPBQ performance track; run non-R4 normal MMA instead. Requested R4 knobs: $($active -join ', ')."
  }
}

function Invoke-Adb {
  param(
    [string[]] $SerialArgs = @(),
    [Parameter(Mandatory = $true)][string[]] $Args,
    [switch] $AllowFail
  )
  $oldErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $out = & adb @SerialArgs @Args 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $oldErrorActionPreference
  if ($out) {
    $out | ForEach-Object { Write-Host $_ }
  }
  if ($code -ne 0 -and -not $AllowFail) {
    throw "adb failed ($code): adb $($SerialArgs -join ' ') $($Args -join ' ')"
  }
  return $out
}

function Invoke-AdbCapture {
  param(
    [string[]] $SerialArgs = @(),
    [Parameter(Mandatory = $true)][string[]] $Args,
    [switch] $AllowFail
  )
  $oldErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $out = & adb @SerialArgs @Args 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $oldErrorActionPreference
  if ($code -ne 0 -and -not $AllowFail) {
    throw "adb failed ($code): adb $($SerialArgs -join ' ') $($Args -join ' ')"
  }
  return ($out -join "`n").Trim()
}

function Push-IfChanged {
  param(
    [string[]] $SerialArgs = @(),
    [Parameter(Mandatory = $true)][string] $LocalPath,
    [Parameter(Mandatory = $true)][string] $RemotePath,
    [switch] $Always
  )
  if (-not (Test-Path -LiteralPath $LocalPath)) {
    throw "missing local file: $LocalPath"
  }

  $localSize = (Get-Item -LiteralPath $LocalPath).Length
  $remoteSize = -1L
  if (-not $Always) {
    $remoteText = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "stat -c %s '$RemotePath' 2>/dev/null || echo 0") -AllowFail
    [void][long]::TryParse($remoteText.Trim(), [ref]$remoteSize)
  }

  if ($Always -or $remoteSize -ne $localSize) {
    Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $LocalPath, $RemotePath) | Out-Null
  } else {
    Write-Host "Remote file already present: $(Split-Path -Leaf $LocalPath) ($remoteSize bytes)"
  }
}

function Write-JsonFile {
  param(
    [Parameter(Mandatory = $true)] $Object,
    [Parameter(Mandatory = $true)][string] $Path,
    [int] $Depth = 16
  )
  $parent = Split-Path -Parent $Path
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $json = $Object | ConvertTo-Json -Depth $Depth
  [System.IO.File]::WriteAllText($Path, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}

function ConvertTo-WslPath {
  param([Parameter(Mandatory = $true)][string] $WindowsPath)
  $full = (Resolve-Path -LiteralPath $WindowsPath).Path
  if ($full -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "cannot convert path to WSL form: $full"
  }
  $drive = $Matches[1].ToLowerInvariant()
  $rest = $Matches[2] -replace '\\', '/'
  return "/mnt/$drive/$rest"
}

function Ensure-LocalHmxFp16Model {
  param([string] $ModelPath = "")

  $experimentRoot = Get-ExperimentRoot
  $localModel = Join-Path $experimentRoot "models\qwen2.5-1.5b-instruct.f16-hmx.gguf"
  $envModel = [Environment]::GetEnvironmentVariable("SNAPDRAGON_NPU_FP16_MODEL")

  # Portability note (2026-07-27): the original experiment searched a
  # workspace-specific artifacts directory and an E: drive. Those legacy
  # locations are intentionally kept only as comments for maintainers:
  # $hmxSource = Join-Path (Get-RepoRoot) "artifacts\llm_gui_v73_qwen15b\models\qwen2.5-1.5b-instruct.f16-hmx.gguf"
  # $rawSource = "<legacy-absolute-model-path>\qwen2.5-1.5b-instruct.f16.gguf"
  # A clean clone now uses -ModelPath, SNAPDRAGON_NPU_FP16_MODEL, or models/.
  $hmxSource = if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath
  } elseif (-not [string]::IsNullOrWhiteSpace($envModel)) {
    $envModel
  } else {
    $localModel
  }

  if (-not (Test-Path -LiteralPath $hmxSource -PathType Leaf)) {
    throw "missing HMX FP16 model: $hmxSource. Pass -ModelPath, set SNAPDRAGON_NPU_FP16_MODEL, or place qwen2.5-1.5b-instruct.f16-hmx.gguf under projects/pure-fp16/models/."
  }
  $hmxSource = (Resolve-Path -LiteralPath $hmxSource).Path
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localModel) | Out-Null

  $manifest = [ordered]@{
    model = Split-Path -Leaf $hmxSource
    local_path = $hmxSource
    local_size = (Get-Item -LiteralPath $hmxSource).Length
    hmx_layout_source = $hmxSource
    hmx_layout_source_size = (Get-Item -LiteralPath $hmxSource).Length
    raw_fp16_source = $null
    raw_fp16_source_exists = $false
    raw_fp16_source_size = $null
    note = "HMX layout changes storage order for the W16A32 kernel only; values remain FP16 and no INT4/INT8 per-group quantization is applied."
  }
  Write-JsonFile -Object $manifest -Path (Join-Path $experimentRoot "models\model_manifest.json")
  return $hmxSource
}

function Get-ApiKey {
  param(
    [string[]] $SerialArgs = @(),
    [string] $AppDir = "/data/local/tmp/llama-npu-chat"
  )
  $key = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh show-key 2>/dev/null") -AllowFail
  $key = ($key -replace "`r", "").Trim()
  if ([string]::IsNullOrWhiteSpace($key) -or $key.StartsWith("ERROR:")) {
    return $null
  }
  return $key
}

function Wait-ServerHealth {
  param(
    [int] $Port = 8080,
    [int] $TimeoutSec = 120
  )
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  $last = ""
  while ((Get-Date) -lt $deadline) {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & curl.exe -sS --noproxy "*" --max-time 3 "http://127.0.0.1:$Port/health" 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    $last = ($out -join "`n")
    if ($code -eq 0 -and $last -match '"status"\s*:\s*"ok"') {
      return $last
    }
    Start-Sleep -Seconds 2
  }
  throw "llama-server did not become healthy on port $Port within $TimeoutSec seconds. Last output: $last"
}

function Invoke-ChatCompletion {
  param(
    [Parameter(Mandatory = $true)] $Payload,
    [Parameter(Mandatory = $true)][string] $RequestPath,
    [Parameter(Mandatory = $true)][string] $ResponsePath,
    [int] $Port = 8080,
    [string] $ApiKey = "",
    [int] $TimeoutSec = 900,
    [int] $RetryCount = 4,
    [int] $RetryDelaySec = 5
  )
  Write-JsonFile -Object $Payload -Path $RequestPath -Depth 32
  $headers = @("-H", "Content-Type: application/json")
  if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $headers += @("-H", "Authorization: Bearer $ApiKey")
  }
  $url = "http://127.0.0.1:$Port/v1/chat/completions"
  $text = ""
  $code = 0
  $debugPath = "$ResponsePath.curl_debug.txt"
  [System.IO.File]::WriteAllText(
    $debugPath,
    "url=$url auth_len=$($(if ([string]::IsNullOrWhiteSpace($ApiKey)) { 0 } else { $ApiKey.Length })) retry_count=$RetryCount`n",
    [System.Text.UTF8Encoding]::new($false))
  for ($attempt = 1; $attempt -le [Math]::Max(1, $RetryCount); $attempt++) {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $body = & curl.exe -sS --noproxy "*" --max-time $TimeoutSec @headers --data-binary "@$RequestPath" $url 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    Add-Content -LiteralPath $debugPath -Encoding UTF8 -Value "attempt=$attempt code=$code"
    $text = ($body -join "`n")
    [System.IO.File]::WriteAllText($ResponsePath, $text + "`n", [System.Text.UTF8Encoding]::new($false))
    if ($code -eq 0) {
      break
    }
    if ($attempt -lt $RetryCount) {
      # The Android llama-server occasionally drops the first POST after HTP
      # warmup while /health stays OK. Retry only the same saved request body.
      Start-Sleep -Seconds $RetryDelaySec
      Wait-ServerHealth -Port $Port -TimeoutSec 30 | Out-Null
    }
  }
  if ($code -ne 0) {
    throw "curl failed ($code) for $url after $RetryCount attempts; response saved to $ResponsePath"
  }
  return ($text | ConvertFrom-Json)
}

function Get-ResponseContent {
  param($Response)
  try {
    return [string]$Response.choices[0].message.content
  } catch {
    return ""
  }
}

function Get-DeviceLogSize {
  param(
    [string[]] $SerialArgs = @(),
    [string] $AppDir = "/data/local/tmp/llama-npu-chat"
  )
  $text = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && wc -c < server.log 2>/dev/null || echo 0") -AllowFail
  $size = 0L
  [void][long]::TryParse(($text -replace "[^0-9]", ""), [ref]$size)
  return $size
}

function Collect-DeviceLogSince {
  param(
    [string[]] $SerialArgs = @(),
    [string] $AppDir = "/data/local/tmp/llama-npu-chat",
    [long] $Offset = 0,
    [Parameter(Mandatory = $true)][string] $OutPath
  )
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutPath) | Out-Null
  $start = $Offset + 1
  & adb @SerialArgs exec-out "cd $AppDir && tail -c +$start server.log" > $OutPath
  if ($LASTEXITCODE -ne 0) {
    throw "adb exec-out log capture failed"
  }
}
