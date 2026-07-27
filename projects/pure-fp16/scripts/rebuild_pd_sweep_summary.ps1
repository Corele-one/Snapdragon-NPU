param(
  [string] $OutRoot = "",
  [int] $Repeats = 3
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  $OutRoot = Get-ChildItem (Join-Path $ExperimentRoot "sequence_benchmarks") -Directory |
    Where-Object { $_.Name -like "pure_fp16_sequence_sweep_*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
}
if (-not (Test-Path -LiteralPath $OutRoot)) {
  throw "OutRoot not found: $OutRoot"
}

$RequestsDir = Join-Path $OutRoot "requests"
$ResponsesDir = Join-Path $OutRoot "responses"
$ProfilingDir = Join-Path $OutRoot "profiling"

$cases = @(
  @{ name = "prefill_64_decode_16"; prefill = 64; decode = 16 },
  @{ name = "prefill_256_decode_16"; prefill = 256; decode = 16 },
  @{ name = "prefill_768_decode_16"; prefill = 768; decode = 16 },
  @{ name = "prefill_256_decode_64"; prefill = 256; decode = 64 },
  @{ name = "long_p1024_d16"; prefill = 1024; decode = 16 },
  @{ name = "long_p1024_d128"; prefill = 1024; decode = 128 },
  @{ name = "long_p1536_d64"; prefill = 1536; decode = 64 },
  @{ name = "long_p1792_d256"; prefill = 1792; decode = 256 }
)

function Read-TimingField {
  param($Response, [string] $Name)
  if ($Response.timings -and ($null -ne $Response.timings.$Name)) {
    return [double] $Response.timings.$Name
  }
  return [double] 0
}

function Read-MatmulStageSummary {
  param([string] $StageCsv)
  $summary = [ordered]@{
    total_us = 0.0
    hmx_mma_us = 0.0
    weight_load_us = 0.0
    weight_dequant_us = 0.0
  }
  if (-not (Test-Path -LiteralPath $StageCsv)) {
    return $summary
  }

  # Stream the large stage CSVs instead of Import-Csv loading multi-GB traces
  # into memory. The parser writes simple comma-separated fields for the columns
  # used below, so indexed splitting is sufficient here.
  $reader = [System.IO.StreamReader]::new($StageCsv)
  try {
    $header = $reader.ReadLine()
    if ([string]::IsNullOrWhiteSpace($header)) {
      return $summary
    }
    $cols = $header.Split(',')
    $opIdx = [Array]::IndexOf($cols, "op")
    $stageIdx = [Array]::IndexOf($cols, "stage")
    $durIdx = [Array]::IndexOf($cols, "dur_us")
    if ($opIdx -lt 0 -or $stageIdx -lt 0 -or $durIdx -lt 0) {
      return $summary
    }

    while (($line = $reader.ReadLine()) -ne $null) {
      $parts = $line.Split(',')
      if ($parts.Length -le [Math]::Max($durIdx, [Math]::Max($opIdx, $stageIdx))) {
        continue
      }
      $op = $parts[$opIdx]
      if ($op -ne "matmul_f16" -and $op -ne "matmul_w16a32") {
        continue
      }
      $dur = 0.0
      if (-not [double]::TryParse($parts[$durIdx], [ref] $dur)) {
        continue
      }
      $summary.total_us += $dur
      switch ($parts[$stageIdx]) {
        "hmx_mma" { $summary.hmx_mma_us += $dur }
        "weight_hvx_load" { $summary.weight_load_us += $dur }
        "weight_hvx_dequant" { $summary.weight_dequant_us += $dur }
      }
    }
  } finally {
    $reader.Close()
  }
  return $summary
}

$rawRows = @()
foreach ($case in $cases) {
  for ($i = 1; $i -le $Repeats; $i++) {
    $runName = "$($case.name)_r$("{0:D2}" -f $i)"
    $respPath = Join-Path $ResponsesDir "$runName.json"
    if (-not (Test-Path -LiteralPath $respPath)) {
      throw "missing response: $respPath"
    }
    $response = Get-Content -LiteralPath $respPath -Raw | ConvertFrom-Json
    $rawRows += [pscustomobject][ordered]@{
      case = $case.name
      repeat = $i
      target_prefill_tokens = $case.prefill
      max_tokens = $case.decode
      actual_prompt_tokens = $response.usage.prompt_tokens
      actual_completion_tokens = $response.usage.completion_tokens
      prompt_tokens_per_s = Read-TimingField -Response $response -Name "prompt_per_second"
      decode_tokens_per_s = Read-TimingField -Response $response -Name "predicted_per_second"
      response = $respPath
    }
  }
}
$rawRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutRoot "raw_no_trace_runs.csv")

$traceRows = @()
foreach ($case in $cases) {
  $runName = "trace_$($case.name)"
  $respPath = Join-Path $ResponsesDir "$runName.json"
  $outDir = Join-Path $ProfilingDir $runName
  $stageCsv = Join-Path $outDir "llm_trace_stage_events.csv"
  if (-not (Test-Path -LiteralPath $respPath)) {
    throw "missing trace response: $respPath"
  }
  $response = Get-Content -LiteralPath $respPath -Raw | ConvertFrom-Json
  $stageSummary = Read-MatmulStageSummary -StageCsv $stageCsv
  $matmulTotal = [double] $stageSummary.total_us
  $traceRows += [pscustomobject][ordered]@{
    case = $case.name
    actual_prompt_tokens = $response.usage.prompt_tokens
    actual_completion_tokens = $response.usage.completion_tokens
    trace_dir = $outDir
    trace_events_csv = Join-Path $outDir "llm_trace_events.csv"
    trace_stage_events_csv = $stageCsv
    trace_ntff = Join-Path $outDir "traces\llm_inference_pure_fp16.ntff"
    matmul_stage_total_us = [Math]::Round($matmulTotal, 3)
    matmul_hmx_mma_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $stageSummary.hmx_mma_us / $matmulTotal, 4) } else { 0 }
    matmul_weight_load_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $stageSummary.weight_load_us / $matmulTotal, 4) } else { 0 }
    matmul_weight_dequant_pct = if ($matmulTotal -gt 0) { [Math]::Round(100.0 * $stageSummary.weight_dequant_us / $matmulTotal, 4) } else { 0 }
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
Write-JsonFile -Object $summaryRows -Path (Join-Path $OutRoot "summary_sequence_breakdown.json") -Depth 12

$md = @()
$md += "# Pure FP16 Sequence Length Sweep"
$md += ""
$md += "Runtime: isolated pure_fp16, HMX-layout FP16 weights, no INT4/INT8 per-group weight dequantization."
$md += ""
$md += "| case | P:D | runs | actual prompt | actual decode | prefill tok/s | decode tok/s | HMX MMA % | weight load % | dequant % | trace |"
$md += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
foreach ($row in $summaryRows) {
  $traceRel = Resolve-Path -LiteralPath $row.trace_dir -Relative
  $md += "| $($row.case) | $($row.P_D) | $($row.runs) | $($row.actual_prompt_tokens_mean) | $($row.actual_decode_tokens_mean) | $($row.prefill_tok_s_mean) | $($row.decode_tok_s_mean) | $($row.matmul_hmx_mma_pct) | $($row.matmul_weight_load_pct) | $($row.matmul_weight_dequant_pct) | ``$traceRel`` |"
}
$md += ""
$md += "Artifacts:"
$md += '- Raw no-trace runs: `raw_no_trace_runs.csv`'
$md += '- One detailed trace per case: `profiling/trace_<case>/`'
$md += '- Each trace directory contains `llm_trace_events.csv`, `llm_trace_stage_events.csv`, `llm_trace_summary.json`, `llm_trace_breakdown.md`, `.ntff`, and `.perfetto.json`.'
$md += '- Acceptance check: `matmul_weight_dequant_pct` must remain `0`, proving this route did not enter `weight_hvx_dequant`.'
$md -join "`n" | Set-Content -LiteralPath (Join-Path $OutRoot "summary_sequence_breakdown.md") -Encoding UTF8

Write-Host "Rebuilt PD summary under $OutRoot"
