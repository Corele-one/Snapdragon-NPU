param(
  [int] $Port = 8080,
  [string] $Serial = "",
  [string] $AppDir = "/data/local/tmp/llama-npu-chat",
  [string] $CaseName = "prefill_512_decode_16",
  [int] $TargetPrefill = 512,
  [int] $Decode = 16,
  [int] $Batch = 0,
  [int] $Ubatch = 0,
  [int] $Threads = 4
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
$RepoRoot = Get-RepoRoot
$SerialArgs = Get-AdbSerialArgs $Serial
if ($TargetPrefill -gt 0 -and $TargetPrefill -lt 512) {
  throw "Pure FP16 trace prefill performance tests require TargetPrefill >= 512; got $TargetPrefill"
}
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$OutDir = Join-Path $ExperimentRoot "profiling_traces\${CaseName}_$stamp"
$LogPath = Join-Path $OutDir "server_trace.log"
New-Item -ItemType Directory -Force -Path $OutDir, (Join-Path $ExperimentRoot "requests"), (Join-Path $ExperimentRoot "responses") | Out-Null

$switchBatch = if ($Batch -gt 0) { $Batch } else { 512 }
$switchUbatch = if ($Ubatch -gt 0) { $Ubatch } else { 512 }
& "$PSScriptRoot\switch_llm_inference_route.ps1" -Mode pure_fp16 -Port $Port -Serial $Serial -AppDir $AppDir -Batch $switchBatch -Ubatch $switchUbatch -Threads $Threads -Trace -DetailedTrace
if ($Batch -gt 0 -or $Ubatch -gt 0) {
  $runBatch = $switchBatch
  $runUbatch = $switchUbatch
  # Restart after switch so trace runs can pin the exact scheduler split being
  # measured; otherwise run_server.sh defaults would hide batch/ubatch effects.
  Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh stop") -AllowFail | Out-Null
  Invoke-Adb -SerialArgs $SerialArgs -Args @(
    "shell",
    "cd $AppDir && ./run_server.sh start qwen2.5-1.5b-instruct.f16-hmx.gguf pure_fp16 1 1 $Port $runBatch $runUbatch $Threads 4096 2048"
  ) | Out-Null
  Invoke-Adb -SerialArgs $SerialArgs -Args @("forward", "tcp:$Port", "tcp:$Port") | Out-Null
  Wait-ServerHealth -Port $Port -TimeoutSec 180 | Out-Null
}
$apiKey = Get-ApiKey -SerialArgs $SerialArgs -AppDir $AppDir
$offset = Get-DeviceLogSize -SerialArgs $SerialArgs -AppDir $AppDir

if ($TargetPrefill -gt 0) {
  $sentence = "mobile npu profiling prefill workload keeps attention and feed forward layers active while preserving a deterministic prompt. "
  $reps = [Math]::Max(1, [int]($TargetPrefill / 12))
  $prompt = "This is a profiling-only prompt. $($sentence * $reps)`nRepeat the word alpha until stopped. Do not explain. The generation cap is $Decode tokens."
} else {
  $prompt = "Return exactly SAFE_VALUE=314159 and then stop."
}

$payload = [ordered]@{
  messages = @(@{ role = "user"; content = $prompt })
  stream = $false
  cache_prompt = $false
  npu_mode = "pure_fp16"
  temperature = 0
  top_k = 1
  max_tokens = $Decode
  timings_per_token = $true
}
$response = Invoke-ChatCompletion -Payload $payload -RequestPath (Join-Path $ExperimentRoot "requests\trace_${CaseName}_$stamp.json") -ResponsePath (Join-Path $ExperimentRoot "responses\trace_${CaseName}_$stamp.json") -Port $Port -ApiKey $apiKey -TimeoutSec 600
Start-Sleep -Seconds 1
Collect-DeviceLogSince -SerialArgs $SerialArgs -AppDir $AppDir -Offset $offset -OutPath $LogPath

$parseScript = Join-Path $RepoRoot "tools\parse_llm_inference_trace.py"
& python $parseScript --log $LogPath --out-dir $OutDir | Tee-Object -FilePath (Join-Path $OutDir "parse_stdout.txt")
if ($LASTEXITCODE -ne 0) {
  throw "trace parser failed"
}

$events = Join-Path $OutDir "llm_trace_events.csv"
$stages = Join-Path $OutDir "llm_trace_stage_events.csv"
$eventText = if (Test-Path -LiteralPath $events) { Get-Content -LiteralPath $events -Raw } else { "" }
$stageText = if (Test-Path -LiteralPath $stages) { Get-Content -LiteralPath $stages -Raw } else { "" }
$hasMatmul = ($eventText -match "matmul_f16|matmul_w16a32") -or ($stageText -match "matmul_f16|matmul_w16a32")
$hasDequant = $stageText -match "weight_hvx_dequant"

$validation = [ordered]@{
  created_at = (Get-Date).ToString("o")
  case = $CaseName
  target_prefill = $TargetPrefill
  max_tokens = $Decode
  batch = $Batch
  ubatch = $Ubatch
  output = (Get-ResponseContent $response)
  actual_prompt_tokens = $response.usage.prompt_tokens
  actual_completion_tokens = $response.usage.completion_tokens
  prefill_tok_s = $response.timings.prompt_per_second
  decode_tok_s = $response.timings.predicted_per_second
  out_dir = $OutDir
  log = $LogPath
  has_fp16_matmul = [bool]$hasMatmul
  has_weight_hvx_dequant = [bool]$hasDequant
  passed = [bool]($hasMatmul -and -not $hasDequant)
}
Write-JsonFile -Object $validation -Path (Join-Path $OutDir "trace_validation.json") -Depth 12
if (-not $validation.passed) {
  Write-Host "Trace validation failed. See $OutDir"
  exit 1
}
Write-Host "Trace validation passed. Output: $OutDir"
