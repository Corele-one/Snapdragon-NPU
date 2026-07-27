param(
  [ValidateSet("pure_fp16", "iq4", "lpbq_int8")]
  [string] $Mode = "pure_fp16",

  [int] $Port = 8080,
  [string] $Serial = "",
  [string] $AppDir = "/data/local/tmp/llama-npu-chat",
  [int] $Batch = 512,
  [int] $Ubatch = 512,
  [int] $Threads = 4,
  [switch] $LpbqFullV6Weight,
  [switch] $LpbqFullV6NonR4,
  [string] $LpbqFullV6NonR4AllowList = "",
  [int] $LpbqV6FullGroupTiles = 16,
  [switch] $LpbqR4UseFullV6WeightFd,
  [switch] $LpbqR4CompactFullU8SafeAB,
  [int] $LpbqR4FullV6MinM = 32,
  [int] $LpbqR4FullV6SmallMMax = 256,
  [switch] $ForceR4FullU8Safe,
  [switch] $ForceR4FullU8IgnoreList,
  [string] $R4FullU8SafeList = "",
  [int] $LpbqR4StructuredFwhtSmallMMax = 4,
  [switch] $LpbqR4HmxDenseFp16Sidecar,
  [switch] $NoSwitch,
  [switch] $CpuFallback,
  [switch] $NoWarmup
)

. "$PSScriptRoot\common.ps1"

if ($Mode -eq "lpbq_int8") {
  $r4Requests = @()
  if ($LpbqR4UseFullV6WeightFd) { $r4Requests += "-LpbqR4UseFullV6WeightFd" }
  if ($LpbqR4CompactFullU8SafeAB) { $r4Requests += "-LpbqR4CompactFullU8SafeAB" }
  if ($ForceR4FullU8Safe) { $r4Requests += "-ForceR4FullU8Safe" }
  if ($ForceR4FullU8IgnoreList) { $r4Requests += "-ForceR4FullU8IgnoreList" }
  if (-not [string]::IsNullOrWhiteSpace($R4FullU8SafeList)) { $r4Requests += "-R4FullU8SafeList" }
  if ($LpbqR4FullV6MinM -ne 32) { $r4Requests += "-LpbqR4FullV6MinM" }
  if ($LpbqR4FullV6SmallMMax -ne 256) { $r4Requests += "-LpbqR4FullV6SmallMMax" }
  if ($LpbqR4StructuredFwhtSmallMMax -ne 4) { $r4Requests += "-LpbqR4StructuredFwhtSmallMMax" }
  if ($LpbqR4HmxDenseFp16Sidecar) { $r4Requests += "-LpbqR4HmxDenseFp16Sidecar" }
  # R4/FWHT is hard-cancelled for LPBQ performance work; historical R4 summary
  # fields below stay only so old validation files remain understandable.
  Assert-LpbqR4PerformancePathCancelled -RequestedKnobs @($r4Requests)
}

$ExperimentRoot = Get-ExperimentRoot
$SerialArgs = Get-AdbSerialArgs $Serial
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$summaryPath = Join-Path $ExperimentRoot "validation\smoke_${Mode}_$stamp.json"
New-Item -ItemType Directory -Force -Path (Join-Path $ExperimentRoot "requests"), (Join-Path $ExperimentRoot "responses"), (Join-Path $ExperimentRoot "validation") | Out-Null

if (-not $NoSwitch) {
  # LPBQ signed full-U8 bring-up can spend minutes in llama.cpp's empty-run
  # warmup. The smoke validates real request correctness, so skip only that
  # startup warmup by default for LPBQ while leaving pure-FP16/IQ4 unchanged.
  $skipWarmupForSmoke = $NoWarmup -or ($Mode -eq "lpbq_int8")
  & "$PSScriptRoot\switch_llm_inference_route.ps1" -Mode $Mode -Port $Port -Serial $Serial -AppDir $AppDir -Batch $Batch -Ubatch $Ubatch -Threads $Threads -LpbqFullV6Weight:$LpbqFullV6Weight -LpbqFullV6NonR4:$LpbqFullV6NonR4 -LpbqFullV6NonR4AllowList $LpbqFullV6NonR4AllowList -LpbqV6FullGroupTiles $LpbqV6FullGroupTiles -LpbqR4UseFullV6WeightFd:$LpbqR4UseFullV6WeightFd -LpbqR4CompactFullU8SafeAB:$LpbqR4CompactFullU8SafeAB -LpbqR4FullV6MinM $LpbqR4FullV6MinM -LpbqR4FullV6SmallMMax $LpbqR4FullV6SmallMMax -ForceR4FullU8Safe:$ForceR4FullU8Safe -ForceR4FullU8IgnoreList:$ForceR4FullU8IgnoreList -R4FullU8SafeList $R4FullU8SafeList -LpbqR4StructuredFwhtSmallMMax $LpbqR4StructuredFwhtSmallMMax -LpbqR4HmxDenseFp16Sidecar:$LpbqR4HmxDenseFp16Sidecar -NoWarmup:$skipWarmupForSmoke
}

if ($CpuFallback) {
  # CPU fallback is only intended for the legacy FP16/IQ4 smoke checks; LPBQ
  # needs the NPU sidecar path to validate the deployment being optimized.
  if ($Mode -eq "lpbq_int8") {
    throw "CpuFallback is not supported for lpbq_int8 smoke; run without -CpuFallback to test the NPU LPBQ path."
  }
  $model = if ($Mode -eq "pure_fp16") { "qwen2.5-1.5b-instruct.f16-hmx.gguf" } else { "qwen2.5-1.5b-instruct.iq4_nl+q8_0-hmx.gguf" }
  $npuMode = if ($Mode -eq "pure_fp16") { "pure_fp16" } else { "baseline" }
  Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh stop") -AllowFail | Out-Null
  Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && SKIP_HTP_OPS=1 ./run_server.sh start '$model' '$npuMode' 0 0 '$Port' '$Batch' '$Ubatch' '$Threads'") | Out-Null
  Invoke-Adb -SerialArgs $SerialArgs -Args @("forward", "tcp:$Port", "tcp:$Port") | Out-Null
  Wait-ServerHealth -Port $Port -TimeoutSec 180 | Out-Null
}

$apiKey = Get-ApiKey -SerialArgs $SerialArgs -AppDir $AppDir
$tests = @(
  [ordered]@{
    name = "math_short"
    max_tokens = 16
    prompt = "Return only the number 12. No words, no punctuation. Question: 7+5=?"
    check = "contains_12"
  },
  [ordered]@{
    name = "safe_value"
    max_tokens = 32
    prompt = "Return exactly SAFE_VALUE=314159 and nothing else."
    check = "safe_value"
  },
  [ordered]@{
    name = "cn_npu"
    max_tokens = 96
    # Keep the script ASCII-safe for Windows PowerShell, but require CJK output
    # below so this still validates the Chinese-answer smoke requirement.
    prompt = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("6K+355So5LiA5Y+l566A55+t5Lit5paH6Kej6YeK5LuA5LmI5pivIE5QVe+8jOW/hemhu+WMheWQqyBOUFXvvIzkuI3opoHph43lpI3jgII="))
    check = "cn_npu"
  }
)

function ConvertTo-SmokeContentPreview {
  param(
    [AllowNull()][string] $Text,
    [int] $MaxChars = 160
  )
  if ($null -eq $Text) {
    return ""
  }
  $limit = [Math]::Min($Text.Length, $MaxChars)
  $builder = [System.Text.StringBuilder]::new()
  for ($i = 0; $i -lt $limit; $i++) {
    $ch = $Text[$i]
    $code = [int][char]$ch
    if ($code -ge 32 -and $code -le 126 -and $code -ne 34 -and $code -ne 92) {
      [void]$builder.Append($ch)
    } elseif ($code -eq 34) {
      [void]$builder.Append("'")
    } elseif ($code -eq 92) {
      [void]$builder.Append("/")
    } else {
      [void]$builder.Append(("\u{0:X4}" -f $code))
    }
  }
  if ($Text.Length -gt $MaxChars) {
    [void]$builder.Append("...")
  }
  return $builder.ToString()
}

$results = @()
foreach ($test in $tests) {
  $payload = [ordered]@{
    messages = @(@{ role = "user"; content = $test.prompt })
    stream = $false
    cache_prompt = $false
    npu_mode = $Mode
    temperature = 0
    top_k = 1
    max_tokens = $test.max_tokens
    timings_per_token = $true
  }
  $reqPath = Join-Path $ExperimentRoot "requests\smoke_$($test.name)_$stamp.json"
  $respPath = Join-Path $ExperimentRoot "responses\smoke_$($test.name)_$stamp.json"
  $response = Invoke-ChatCompletion -Payload $payload -RequestPath $reqPath -ResponsePath $respPath -Port $Port -ApiKey $apiKey -TimeoutSec 300
  $content = Get-ResponseContent $response
  $isPass = $false
  $reason = ""
  if ($test.check -eq "contains_12") {
    $isPass = $content -match "(^|[^0-9])12([^0-9]|$)"
    $reason = if ($isPass) { "contains 12" } else { "missing standalone 12" }
  } elseif ($test.check -eq "safe_value") {
    $isPass = $content -match "SAFE_VALUE\s*=\s*314159"
    $reason = if ($isPass) { "contains sentinel" } else { "missing sentinel" }
  } else {
    $noReplacement = $content -notmatch [char]0xfffd
    $noLongRepeat = $content -notmatch "(.{2,24})\1{6,}"
    $hasNpu = $content -match "NPU"
    $hasCjk = $content -match "[\u4e00-\u9fff]"
    $isPass = $noReplacement -and $noLongRepeat -and $hasNpu -and $hasCjk -and $content.Length -gt 0
    $reason = "utf8=$noReplacement repeat_ok=$noLongRepeat keyword=$hasNpu cjk=$hasCjk"
  }
  $results += [ordered]@{
    name = $test.name
    passed = [bool]$isPass
    reason = $reason
    # 2026-06-15: keep raw response text in the per-request response file only.
    # Some CJK console/rendering paths can make a copied raw content field
    # difficult to parse later; the summary stores a JSON-safe preview instead.
    # content = $content
    content_preview = ConvertTo-SmokeContentPreview -Text $content
    content_length = $content.Length
    request = $reqPath
    response = $respPath
    prompt_tokens = $response.usage.prompt_tokens
    completion_tokens = $response.usage.completion_tokens
    prompt_per_second = $response.timings.prompt_per_second
    predicted_per_second = $response.timings.predicted_per_second
  }
}

$summary = [ordered]@{
  created_at = (Get-Date).ToString("o")
  mode = $Mode
  cpu_fallback = [bool]$CpuFallback
  lpbq_full_v6_weight = [bool]($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight)
  lpbq_full_v6_non_r4 = [bool]($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight -and $LpbqFullV6NonR4)
  lpbq_full_v6_non_r4_allow_list = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight -and $LpbqFullV6NonR4 -and -not [string]::IsNullOrWhiteSpace($LpbqFullV6NonR4AllowList)) { $LpbqFullV6NonR4AllowList } else { $null }
  lpbq_v6_full_group_tiles = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqV6FullGroupTiles } else { $null }
  lpbq_r4_use_full_v6_weight_fd = [bool]($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight -and $LpbqR4UseFullV6WeightFd)
  lpbq_r4_compact_full_u8_safe_ab = [bool]($Mode -eq "lpbq_int8" -and $LpbqR4CompactFullU8SafeAB)
  lpbq_r4_full_v6_min_m = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqR4FullV6MinM } else { $null }
  lpbq_r4_full_v6_small_m_max = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqR4FullV6SmallMMax } else { $null }
  lpbq_force_r4_full_u8_safe = [bool]($Mode -eq "lpbq_int8" -and $ForceR4FullU8Safe)
  lpbq_force_r4_full_u8_ignore_list = [bool]($Mode -eq "lpbq_int8" -and $ForceR4FullU8Safe -and $ForceR4FullU8IgnoreList)
  lpbq_r4_full_u8_safe_list = if ($Mode -eq "lpbq_int8" -and $ForceR4FullU8Safe -and -not [string]::IsNullOrWhiteSpace($R4FullU8SafeList)) { $R4FullU8SafeList } else { $null }
  lpbq_r4_structured_fwht_small_m_max = if ($Mode -eq "lpbq_int8") { $LpbqR4StructuredFwhtSmallMMax } else { $null }
  lpbq_r4_hmx_dense_fp16_sidecar = [bool]($Mode -eq "lpbq_int8" -and $LpbqR4HmxDenseFp16Sidecar)
  quality_policy = if ($Mode -eq "lpbq_int8" -and $LpbqR4HmxDenseFp16Sidecar) { "no-quality/performance-first experiment; do not claim deploy-quality" } else { $null }
  port = $Port
  passed = -not ($results | Where-Object { -not $_.passed })
  results = $results
}
Write-JsonFile -Object $summary -Path $summaryPath -Depth 20
if (-not $summary.passed) {
  Write-Host "Smoke failed. Summary: $summaryPath"
  exit 1
}
Write-Host "Smoke passed. Summary: $summaryPath"
