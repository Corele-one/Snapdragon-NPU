param(
  [int] $Port = 8080,
  [string] $Serial = "",
  [string] $AppDir = "/data/local/tmp/llama-npu-chat",
  [int] $Repeats = 1,
  [string] $OutRoot = "",
  [string[]] $Configs = @(),
  [string] $ConfigList = "",
  [string] $CaseList = "",
  [int] $CooldownSeconds = 0,
  [int] $ThermalCdspMax = -1,
  [int] $ThermalPollSeconds = 15,
  [int] $ThermalTimeoutSec = 300,
  [string] $RequestNpuMode = "pure_fp16",
  [switch] $NoRestart
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
$SerialArgs = Get-AdbSerialArgs $Serial
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  $OutRoot = Join-Path $ExperimentRoot "sequence_benchmarks\batch_tuning_$(Get-Date -Format yyyyMMdd_HHmmss)"
}
New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot "requests"), (Join-Path $OutRoot "responses") | Out-Null

function New-Prompt {
  param([int] $TargetTokens, [int] $DecodeTokens)
  $sentence = "mobile npu profiling prefill workload keeps attention and feed forward layers active while preserving a deterministic prompt. "
  $reps = [Math]::Max(1, [int]($TargetTokens / 12))
  return "This is a profiling-only prompt. $($sentence * $reps)`nRepeat the word alpha until stopped. Do not explain. The generation cap is $DecodeTokens tokens."
}

function Invoke-TuneCase {
  param([string] $Tag, [string] $Name, [int] $Prefill, [int] $Decode, [int] $Repeat)
  $payload = [ordered]@{
    messages = @(@{ role = "user"; content = (New-Prompt -TargetTokens $Prefill -DecodeTokens $Decode) })
    stream = $false
    cache_prompt = $false
    npu_mode = $RequestNpuMode
    temperature = 0
    top_k = 1
    max_tokens = $Decode
    # max_tokens is sufficient for this llama-server chat endpoint. Leaving
    # ignore_eos unset keeps tuning runs comparable to smoke and avoids masking
    # server-side failures as artificial long-generation timeouts.
    timings_per_token = $true
  }
  $apiKey = Get-ApiKey -SerialArgs $SerialArgs -AppDir $AppDir
  $response = Invoke-ChatCompletion -Payload $payload `
    -RequestPath (Join-Path $OutRoot "requests\${Tag}_${Name}.json") `
    -ResponsePath (Join-Path $OutRoot "responses\${Tag}_${Name}.json") `
    -Port $Port -ApiKey $apiKey -TimeoutSec 900
  return [pscustomobject][ordered]@{
    tag = $Tag
    case = $Name
    repeat = $Repeat
    prompt_n = $response.usage.prompt_tokens
    decode_n = $response.usage.completion_tokens
    prefill_tok_s = [double] $response.timings.prompt_per_second
    decode_tok_s = [double] $response.timings.predicted_per_second
  }
}

function Convert-TuneConfig {
  param([string] $Text)
  $parts = $Text.Split(":")
  if ($parts.Count -lt 3 -or $parts.Count -gt 4) {
    throw "invalid -Configs entry '$Text'; use tag:batch:ubatch[:threads]"
  }
  return [pscustomobject]@{
    tag = $parts[0]
    b = [int] $parts[1]
    ub = [int] $parts[2]
    threads = if ($parts.Count -eq 4) { [int] $parts[3] } else { 4 }
  }
}

function Convert-TuneCase {
  param([string] $Text)
  $parts = $Text.Split(":")
  if ($parts.Count -ne 3) {
    throw "invalid -CaseList entry '$Text'; use name:prefill:decode"
  }
  return @{
    name = $parts[0]
    p = [int] $parts[1]
    d = [int] $parts[2]
  }
}

function Get-CdspCoolingValue {
  $text = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "dumpsys thermalservice 2>/dev/null | grep 'mName=cdsp}' | head -1") -AllowFail
  if ($text -match "mValue=([0-9]+)") {
    return [int]$Matches[1]
  }
  return -1
}

function Wait-CdspCooling {
  if ($ThermalCdspMax -lt 0) {
    return
  }
  $deadline = (Get-Date).AddSeconds($ThermalTimeoutSec)
  while ($true) {
    $value = Get-CdspCoolingValue
    Write-Host "cdsp cooling=$value target<=$ThermalCdspMax"
    if ($value -ge 0 -and $value -le $ThermalCdspMax) {
      return
    }
    if ((Get-Date) -ge $deadline) {
      Write-Warning "Timed out waiting for cdsp cooling <= $ThermalCdspMax; continuing with value=$value"
      return
    }
    Start-Sleep -Seconds $ThermalPollSeconds
  }
}

function Get-TuneConfigValue {
  param($Config, [string] $Name)
  if ($Config -is [string]) {
    $Config = Convert-TuneConfig $Config
  }
  if ($Config -is [System.Collections.IDictionary]) {
    return $Config[$Name]
  }
  $prop = $Config.PSObject.Properties[$Name]
  if ($null -eq $prop) {
    throw "tune config is missing '$Name': $Config"
  }
  return $prop.Value
}

if (-not [string]::IsNullOrWhiteSpace($ConfigList)) {
  # `-ConfigList` is the preferred CLI form because PowerShell `-File` can bind
  # trailing string-array items positionally in surprising ways.
  $configTexts = @($ConfigList.Split(",") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
} elseif ($Configs.Count -eq 0) {
  $configTexts = @(
    "b512_ub512_t4:512:512:4",
    "b1024_ub512_t4:1024:512:4",
    "b2048_ub512_t4:2048:512:4",
    "b4096_ub512_t4:4096:512:4",
    "b2048_ub1024_t4:2048:1024:4",
    "b4096_ub1024_t4:4096:1024:4"
  )
} else {
  $configTexts = @($Configs)
}
if (-not [string]::IsNullOrWhiteSpace($CaseList)) {
  $cases = @($CaseList.Split(",") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { Convert-TuneCase $_.Trim() })
} else {
  $cases = @(
    @{ name = "p256_d16"; p = 256; d = 16 },
    @{ name = "p1024_d16"; p = 1024; d = 16 },
    @{ name = "p1792_d16"; p = 1792; d = 16 }
  )
}

$rows = @()
foreach ($cfgText in $configTexts) {
  $cfg = Convert-TuneConfig $cfgText
  $cfgTag = [string]$cfg.tag
  $cfgB = [int]$cfg.b
  $cfgUb = [int]$cfg.ub
  $cfgThreads = [int]$cfg.threads
  Write-Host "Config ${cfgTag}: batch=$cfgB ubatch=$cfgUb threads=$cfgThreads"
  if (-not $NoRestart) {
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh stop") -AllowFail | Out-Null
    # Keep the server start command explicit in the result tags. Pure-FP16
    # prefill is very sensitive to whether the prompt is split across ubatches.
    $start = "cd $AppDir && ./run_server.sh start qwen2.5-1.5b-instruct.f16-hmx.gguf pure_fp16 0 0 $Port $cfgB $cfgUb $cfgThreads 4096 2048"
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", $start) | Out-Null
    Invoke-Adb -SerialArgs $SerialArgs -Args @("forward", "tcp:$Port", "tcp:$Port") | Out-Null
    Wait-ServerHealth -Port $Port -TimeoutSec 180 | Out-Null
  }

  foreach ($case in $cases) {
    for ($i = 1; $i -le $Repeats; $i++) {
      # Pure-FP16 prefill measurements are very sensitive to CDSP thermal
      # cooling. Waiting here prevents later long-prompt cases from being
      # unfairly compared against cooler short-prompt cases.
      Wait-CdspCooling
      $row = Invoke-TuneCase -Tag $cfgTag -Name $($case.name) -Prefill $($case.p) -Decode $($case.d) -Repeat $i
      $rows += $row
      Write-Host ("{0} {1} r{2}: prefill={3:n3} decode={4:n3}" -f $row.tag, $row.case, $row.repeat, $row.prefill_tok_s, $row.decode_tok_s)
      if ($CooldownSeconds -gt 0) {
        Start-Sleep -Seconds $CooldownSeconds
      }
    }
  }
}

$csv = Join-Path $OutRoot "batch_tuning_summary.csv"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
$summary = foreach ($group in ($rows | Group-Object tag, case)) {
  $items = @($group.Group)
  [pscustomobject][ordered]@{
    tag = $items[0].tag
    case = $items[0].case
    runs = $items.Count
    prompt_n_mean = [Math]::Round(($items | Measure-Object -Property prompt_n -Average).Average, 2)
    decode_n_mean = [Math]::Round(($items | Measure-Object -Property decode_n -Average).Average, 2)
    prefill_tok_s_mean = [Math]::Round(($items | Measure-Object -Property prefill_tok_s -Average).Average, 4)
    decode_tok_s_mean = [Math]::Round(($items | Measure-Object -Property decode_tok_s -Average).Average, 4)
  }
}
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "batch_tuning_grouped.csv")
Write-Host "Batch tuning summary: $csv"
