param(
  [int] $Port = 8080,
  [string] $Serial = "",
  [string] $AppDir = "/data/local/tmp/llama-npu-chat",
  [int] $Repeats = 3,
  [string] $OutRoot = "",
  [int] $Batch = 512,
  [int] $Ubatch = 512,
  [int] $Threads = 4,
  [int] $ThermalCdspMax = 1,
  [int] $ThermalPollSeconds = 20,
  [int] $ThermalTimeoutSec = 420,
  [int] $CooldownSeconds = 12,
  [string[]] $CaseName = @(),
  [switch] $SpeedOnly
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
$RepoRoot = Get-RepoRoot
$SerialArgs = Get-AdbSerialArgs $Serial
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  $OutRoot = Join-Path $ExperimentRoot "sequence_benchmarks\pure_fp16_sequence_sweep_$(Get-Date -Format yyyyMMdd_HHmmss)"
}
$RequestsDir = Join-Path $OutRoot "requests"
$ResponsesDir = Join-Path $OutRoot "responses"
$LogsDir = Join-Path $OutRoot "logs"
$ProfilingDir = Join-Path $OutRoot "profiling"
New-Item -ItemType Directory -Force -Path $RequestsDir, $ResponsesDir, $LogsDir, $ProfilingDir | Out-Null

$cases = @(
  # Historical sub-512 cases are intentionally left commented out: they are
  # useful for old artifact comparison only, not for current prefill acceptance.
  # @{ name = "prefill_64_decode_16"; prefill = 64; decode = 16 },
  # @{ name = "prefill_256_decode_16"; prefill = 256; decode = 16 },
  @{ name = "prefill_512_decode_16"; prefill = 512; decode = 16 },
  @{ name = "prefill_512_decode_64"; prefill = 512; decode = 64 },
  @{ name = "prefill_768_decode_16"; prefill = 768; decode = 16 },
  # @{ name = "prefill_256_decode_64"; prefill = 256; decode = 64 },
  @{ name = "long_p1024_d16"; prefill = 1024; decode = 16 },
  @{ name = "long_p1024_d128"; prefill = 1024; decode = 128 },
  @{ name = "long_p1536_d64"; prefill = 1536; decode = 64 },
  @{ name = "long_p1792_d256"; prefill = 1792; decode = 256 }
)

if ($CaseName.Count -gt 0) {
  # Keep batch/ubatch tuning cheap by allowing exact case-name subsets without changing the default 8-case sweep.
  $wantedCases = @{}
  foreach ($name in $CaseName) {
    $wantedCases[$name] = $true
  }
  $cases = @($cases | Where-Object { $wantedCases.ContainsKey($_.name) })
  if ($cases.Count -eq 0) {
    throw "No sequence benchmark cases matched -CaseName: $($CaseName -join ', ')"
  }
}

function New-Prompt {
  param([int] $TargetTokens, [int] $DecodeTokens)
  $sentence = "mobile npu profiling prefill workload keeps attention and feed forward layers active while preserving a deterministic prompt. "
  $reps = [Math]::Max(1, [int]($TargetTokens / 12))
  $filler = $sentence * $reps
  return "This is a profiling-only prompt. $filler`nRepeat the word alpha until stopped. Do not explain. The generation cap is $DecodeTokens tokens."
}

function New-Payload {
  param([int] $TargetTokens, [int] $DecodeTokens)
  return [ordered]@{
    messages = @(@{ role = "user"; content = (New-Prompt -TargetTokens $TargetTokens -DecodeTokens $DecodeTokens) })
    stream = $false
    cache_prompt = $false
    npu_mode = "pure_fp16"
    temperature = 0
    top_k = 1
    max_tokens = $DecodeTokens
    # Do not force ignore_eos on the OpenAI chat endpoint: this fork honors
    # max_tokens reliably without it, and the extra flag made timeout analysis
    # ambiguous while debugging pure-FP16 decode crashes.
    timings_per_token = $true
  }
}

function Read-TimingField {
  param($Response, [string] $Name)
  if ($Response.timings -and ($null -ne $Response.timings.$Name)) {
    return [double]$Response.timings.$Name
  }
  return [double]0
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

function Start-PureFp16Server {
  param([switch] $Trace)
  $traceFlag = if ($Trace) { 1 } else { 0 }
  $detailFlag = if ($Trace) { 1 } else { 0 }
  Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh stop") -AllowFail | Out-Null
  $start = "cd $AppDir && ./run_server.sh start qwen2.5-1.5b-instruct.f16-hmx.gguf pure_fp16 $traceFlag $detailFlag $Port $Batch $Ubatch $Threads 4096 2048"
  Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", $start) | Out-Null
  Invoke-Adb -SerialArgs $SerialArgs -Args @("forward", "tcp:$Port", "tcp:$Port") | Out-Null
  Wait-ServerHealth -Port $Port -TimeoutSec 180 | Out-Null
}

& "$PSScriptRoot\switch_llm_inference_route.ps1" -Mode pure_fp16 -Port $Port -Serial $Serial -AppDir $AppDir
Start-PureFp16Server
$apiKey = Get-ApiKey -SerialArgs $SerialArgs -AppDir $AppDir

$rawRows = @()
foreach ($case in $cases) {
  $payload = New-Payload -TargetTokens $case.prefill -DecodeTokens $case.decode
  for ($i = 1; $i -le $Repeats; $i++) {
    Wait-CdspCooling
    $runName = "$($case.name)_r$("{0:D2}" -f $i)"
    $response = Invoke-ChatCompletion -Payload $payload -RequestPath (Join-Path $RequestsDir "$runName.json") -ResponsePath (Join-Path $ResponsesDir "$runName.json") -Port $Port -ApiKey $apiKey -TimeoutSec 900
    $rawRows += [pscustomobject][ordered]@{
      case = $case.name
      repeat = $i
      target_prefill_tokens = $case.prefill
      max_tokens = $case.decode
      actual_prompt_tokens = $response.usage.prompt_tokens
      actual_completion_tokens = $response.usage.completion_tokens
      prompt_tokens_per_s = Read-TimingField -Response $response -Name "prompt_per_second"
      decode_tokens_per_s = Read-TimingField -Response $response -Name "predicted_per_second"
      response = Join-Path $ResponsesDir "$runName.json"
    }
    if ($CooldownSeconds -gt 0) {
      Start-Sleep -Seconds $CooldownSeconds
    }
  }
}
$rawRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "raw_no_trace_runs.csv")

if ($SpeedOnly) {
  $summaryRows = @()
  foreach ($case in $cases) {
    $rows = @($rawRows | Where-Object { $_.case -eq $case.name })
    $promptMean = ($rows | Measure-Object -Property prompt_tokens_per_s -Average).Average
    $decodeMean = ($rows | Measure-Object -Property decode_tokens_per_s -Average).Average
    $promptTokens = ($rows | Measure-Object -Property actual_prompt_tokens -Average).Average
    $decodeTokens = ($rows | Measure-Object -Property actual_completion_tokens -Average).Average
    $summaryRows += [pscustomobject][ordered]@{
      case = $case.name
      P_D = "$($case.prefill):$($case.decode)"
      runs = $Repeats
      actual_prompt_tokens_mean = [Math]::Round($promptTokens, 2)
      actual_decode_tokens_mean = [Math]::Round($decodeTokens, 2)
      prefill_tok_s_mean = [Math]::Round($promptMean, 4)
      decode_tok_s_mean = [Math]::Round($decodeMean, 4)
      trace_dir = ""
      trace_ntff = ""
      matmul_hmx_mma_pct = ""
      matmul_weight_load_pct = ""
      matmul_weight_dequant_pct = ""
    }
  }
  $summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "summary_sequence_breakdown.csv")
  Write-JsonFile -Object $summaryRows -Path (Join-Path $OutRoot "summary_sequence_breakdown.json") -Depth 12

  $md = @()
  $tick = [char]96
  $md += "# Pure FP16 Sequence Length Sweep (Speed Only)"
  $md += ""
  $md += "Runtime: isolated pure_fp16, no trace, tuned default batch settings from run_server.sh."
  $md += ""
  $md += "| case | P:D | runs | actual prompt | actual decode | prefill tok/s | decode tok/s |"
  $md += "|---|---:|---:|---:|---:|---:|---:|"
  foreach ($row in $summaryRows) {
    $md += "| $($row.case) | $($row.P_D) | $($row.runs) | $($row.actual_prompt_tokens_mean) | $($row.actual_decode_tokens_mean) | $($row.prefill_tok_s_mean) | $($row.decode_tok_s_mean) |"
  }
  $md += ""
  $md += "Detailed no-dequant trace validation is kept in ${tick}profiling_traces/smoke_trace_fixed_20260526_221324${tick} and can be regenerated with ${tick}run_trace_pure_fp16.ps1${tick}."
  $md -join "`n" | Set-Content -LiteralPath (Join-Path $OutRoot "summary_sequence_breakdown.md") -Encoding UTF8

  Write-Host "Speed-only sequence sweep complete. Output: $OutRoot"
  exit 0
}

& "$PSScriptRoot\switch_llm_inference_route.ps1" -Mode pure_fp16 -Port $Port -Serial $Serial -AppDir $AppDir -Trace -DetailedTrace
Start-PureFp16Server -Trace
$apiKey = Get-ApiKey -SerialArgs $SerialArgs -AppDir $AppDir
$parseScript = Join-Path $RepoRoot "tools\parse_llm_inference_trace.py"
$traceRows = @()
foreach ($case in $cases) {
  $payload = New-Payload -TargetTokens $case.prefill -DecodeTokens $case.decode
  Wait-CdspCooling
  $offset = Get-DeviceLogSize -SerialArgs $SerialArgs -AppDir $AppDir
  $runName = "trace_$($case.name)"
  $response = Invoke-ChatCompletion -Payload $payload -RequestPath (Join-Path $RequestsDir "$runName.json") -ResponsePath (Join-Path $ResponsesDir "$runName.json") -Port $Port -ApiKey $apiKey -TimeoutSec 900
  Start-Sleep -Milliseconds 700
  $logPath = Join-Path $LogsDir "$runName.log"
  $outDir = Join-Path $ProfilingDir $runName
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  Collect-DeviceLogSince -SerialArgs $SerialArgs -AppDir $AppDir -Offset $offset -OutPath $logPath
  & python $parseScript --log $logPath --out-dir $outDir | Out-File -Encoding UTF8 (Join-Path $outDir "parse_stdout.txt")
  if ($LASTEXITCODE -ne 0) {
    throw "trace parse failed for $($case.name)"
  }

  $stageCsv = Join-Path $outDir "llm_trace_stage_events.csv"
  $matmulTotal = 0.0
  $hmxTotal = 0.0
  $weightLoadTotal = 0.0
  $dequantTotal = 0.0
  if (Test-Path -LiteralPath $stageCsv) {
    $stageRows = Import-Csv -LiteralPath $stageCsv
    foreach ($row in $stageRows) {
      if ($row.op -match "matmul_f16|matmul_w16a32") {
        $dur = [double]($row.dur_us)
        $matmulTotal += $dur
        if ($row.stage -eq "hmx_mma") { $hmxTotal += $dur }
        if ($row.stage -eq "weight_hvx_load") { $weightLoadTotal += $dur }
        if ($row.stage -eq "weight_hvx_dequant") { $dequantTotal += $dur }
      }
    }
  }
  $traceRows += [pscustomobject][ordered]@{
    case = $case.name
    actual_prompt_tokens = $response.usage.prompt_tokens
    actual_completion_tokens = $response.usage.completion_tokens
    trace_dir = $outDir
    trace_events_csv = Join-Path $outDir "llm_trace_events.csv"
    trace_stage_events_csv = $stageCsv
    trace_ntff = Join-Path $outDir "traces\llm_inference_pure_fp16.ntff"
    matmul_stage_total_us = [Math]::Round($matmulTotal, 3)
    matmul_hmx_mma_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $hmxTotal / $matmulTotal, 4) } else { 0 }
    matmul_weight_load_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $weightLoadTotal / $matmulTotal, 4) } else { 0 }
    matmul_weight_dequant_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $dequantTotal / $matmulTotal, 4) } else { 0 }
  }
}
$traceRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "trace_once_summary.csv")

$summaryRows = @()
foreach ($case in $cases) {
  $rows = @($rawRows | Where-Object { $_.case -eq $case.name })
  $trace = $traceRows | Where-Object { $_.case -eq $case.name } | Select-Object -First 1
  $promptMean = ($rows | Measure-Object -Property prompt_tokens_per_s -Average).Average
  $decodeMean = ($rows | Measure-Object -Property decode_tokens_per_s -Average).Average
  $promptTokens = ($rows | Measure-Object -Property actual_prompt_tokens -Average).Average
  $decodeTokens = ($rows | Measure-Object -Property actual_completion_tokens -Average).Average
  $summaryRows += [pscustomobject][ordered]@{
    case = $case.name
    P_D = "$($case.prefill):$($case.decode)"
    runs = $Repeats
    actual_prompt_tokens_mean = [Math]::Round($promptTokens, 2)
    actual_decode_tokens_mean = [Math]::Round($decodeTokens, 2)
    prefill_tok_s_mean = [Math]::Round($promptMean, 4)
    decode_tok_s_mean = [Math]::Round($decodeMean, 4)
    trace_dir = $trace.trace_dir
    trace_ntff = $trace.trace_ntff
    matmul_hmx_mma_pct = $trace.matmul_hmx_mma_pct
    matmul_weight_load_pct = $trace.matmul_weight_load_pct
    matmul_weight_dequant_pct = $trace.matmul_weight_dequant_pct
  }
}
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "summary_sequence_breakdown.csv")
Write-JsonFile -Object @{ created_at = (Get-Date).ToString("o"); repeats = $Repeats; rows = $summaryRows } -Path (Join-Path $OutRoot "summary_sequence_breakdown.json") -Depth 20

$md = @()
$tick = [char]96
$md += "# Pure FP16 Sequence Breakdown"
$md += ""
$md += "| case | P:D | runs | prompt tokens | decode tokens | prefill tok/s | decode tok/s | HMX MMA % | weight load % | dequant % | trace |"
$md += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
foreach ($row in $summaryRows) {
  $tracePath = $row.trace_dir
  $md += "| $($row.case) | $($row.P_D) | $($row.runs) | $($row.actual_prompt_tokens_mean) | $($row.actual_decode_tokens_mean) | $($row.prefill_tok_s_mean) | $($row.decode_tok_s_mean) | $($row.matmul_hmx_mma_pct) | $($row.matmul_weight_load_pct) | $($row.matmul_weight_dequant_pct) | ${tick}${tracePath}${tick} |"
}
$md += ""
$md += "Acceptance: ${tick}matmul_weight_dequant_pct${tick} must stay ${tick}0${tick} for pure_fp16 traces; trace directories contain ${tick}llm_trace_events.csv${tick}, ${tick}llm_trace_stage_events.csv${tick}, ${tick}llm_trace_summary.json${tick}, ${tick}llm_trace_breakdown.md${tick}, ${tick}.ntff${tick}, and ${tick}.perfetto.json${tick}."
[System.IO.File]::WriteAllText((Join-Path $OutRoot "summary_sequence_breakdown.md"), ($md -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Host "PD sweep complete: $OutRoot"
