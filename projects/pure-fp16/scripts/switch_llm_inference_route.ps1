param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("pure_fp16", "iq4", "lpbq_int8")]
  [string] $Mode,

  [ValidateSet("v73", "v75", "v79")]
  [string] $DspArch = "v73",

  [int] $Port = 8080,
  [string] $Serial = "",
  [string] $AppDir = "/data/local/tmp/llama-npu-chat",
  [int] $Batch = 512,
  [int] $Ubatch = 512,
  [int] $Threads = 4,
  [int] $Context = 4096,
  [int] $Predict = 2048,
  [string] $ModelPath = "",
  [string] $RepoRootOverride = "",
  [switch] $Trace,
  [switch] $DetailedTrace,
  [switch] $LpbqFullV6Weight,
  [switch] $LpbqFullV6NonR4,
  [string] $LpbqFullV6NonR4AllowList = "",
  [int] $LpbqV6FullGroupTiles = 16,
  [switch] $LpbqR4UseFullV6WeightFd,
  [switch] $LpbqR4CompactFullU8SafeAB,
  [switch] $EnableR4,
  [switch] $ForceR4FullU8Safe,
  [switch] $ForceR4FullU8IgnoreList,
  [string] $R4FullU8SafeList = "",
  [int] $LpbqR4FullV6MinM = 32,
  [int] $LpbqR4FullV6SmallMMax = 256,
  [int] $LpbqR4StructuredFwhtSmallMMax = 4,
  [switch] $LpbqR4HmxDenseFp16Sidecar,
  [switch] $NoWarmup
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
$RepoRoot = if ([string]::IsNullOrWhiteSpace($RepoRootOverride)) { Get-RepoRoot } else { (Resolve-Path -LiteralPath $RepoRootOverride).Path }
$DeployDir = Join-Path $ExperimentRoot "deploy"
$SerialArgs = Get-AdbSerialArgs $Serial
$TraceFlag = if ($Trace -or $DetailedTrace) { "1" } else { "0" }
$DetailedFlag = if ($DetailedTrace) { "1" } else { "0" }
$NoWarmupFlag = if ($NoWarmup) { "1" } else { "0" }

if ($Mode -eq "lpbq_int8") {
  $r4Requests = @()
  if ($EnableR4) { $r4Requests += "-EnableR4" }
  if ($LpbqR4UseFullV6WeightFd) { $r4Requests += "-LpbqR4UseFullV6WeightFd" }
  if ($LpbqR4CompactFullU8SafeAB) { $r4Requests += "-LpbqR4CompactFullU8SafeAB" }
  if ($ForceR4FullU8Safe) { $r4Requests += "-ForceR4FullU8Safe" }
  if ($ForceR4FullU8IgnoreList) { $r4Requests += "-ForceR4FullU8IgnoreList" }
  if (-not [string]::IsNullOrWhiteSpace($R4FullU8SafeList)) { $r4Requests += "-R4FullU8SafeList" }
  if ($LpbqR4FullV6MinM -ne 32) { $r4Requests += "-LpbqR4FullV6MinM" }
  if ($LpbqR4FullV6SmallMMax -ne 256) { $r4Requests += "-LpbqR4FullV6SmallMMax" }
  if ($LpbqR4StructuredFwhtSmallMMax -ne 4) { $r4Requests += "-LpbqR4StructuredFwhtSmallMMax" }
  if ($LpbqR4HmxDenseFp16Sidecar) { $r4Requests += "-LpbqR4HmxDenseFp16Sidecar" }
  # 2026-07-03 R4 cancellation: the old opt-in path is intentionally blocked
  # after the FHT roofline check showed it cannot be hidden for ffn_down.
  # Rollback reference: this used to emit LLAMA_NPU_LPBQ_ENABLE_R4_PATH=1 and
  # accept force/full-U8 R4 envs. Keep the env assembly below for rollback
  # context, but reject all explicit R4 requests before touching adb.
  # throw "R4/FWHT path is cancelled for the LPBQ performance track; run non-R4 normal MMA instead."
  Assert-LpbqR4PerformancePathCancelled -RequestedKnobs @($r4Requests)
}

Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "mkdir -p $AppDir $AppDir/cdsp $AppDir/dsp") | Out-Null

$runner = Join-Path $DeployDir "run_server.sh"
if (-not (Test-Path -LiteralPath $runner)) {
  throw "missing runner: $runner"
}

if ($Mode -eq "pure_fp16") {
  $modelPath = Ensure-LocalHmxFp16Model -ModelPath $ModelPath
  $remoteModel = Split-Path -Leaf $modelPath
  $remoteNpuMode = "pure_fp16"
  $runtimeLabel = "isolated-pure-fp16"
  $runtimeFiles = @(
    @{ Local = (Join-Path $DeployDir "run_server.sh"); Remote = "run_server.sh"; Always = $true },
    @{ Local = (Join-Path $DeployDir "llama-server"); Remote = "llama-server"; Always = $true },
    @{ Local = (Join-Path $DeployDir "llama-cli"); Remote = "llama-cli"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml.so"); Remote = "libggml.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-base.so"); Remote = "libggml-base.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-cpu.so"); Remote = "libggml-cpu.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-htp.so"); Remote = "libggml-htp.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libllama.so"); Remote = "libllama.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libhtp_ops.so"); Remote = "libhtp_ops.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libhtp_ops_skel.so"); Remote = "libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "cdsp\libhtp_ops_skel.so"); Remote = "cdsp/libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "dsp\libhtp_ops_skel.so"); Remote = "dsp/libhtp_ops_skel.so"; Always = $true }
  )
} elseif ($Mode -eq "lpbq_int8") {
  $lpbqModelDir = Join-Path $ExperimentRoot "models\lpbq_g16_a8w8"
  $modelPath = Join-Path $lpbqModelDir "qwen2.5-1.5b-ost-g16-lpbq-a8w8.gguf"
  $sidecarPath = Join-Path $lpbqModelDir "sidecars\lpbq_g16_a8w8"
  $r4FwhtV2SidecarPath = Join-Path $lpbqModelDir "sidecars\r4_fwht_v2"
  $r4HmxDenseFp16SidecarPath = Join-Path $lpbqModelDir "sidecars\r4_hmx_dense_fp16_v1"
  if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "missing LPBQ GGUF: $modelPath"
  }
  if (-not (Test-Path -LiteralPath $sidecarPath)) {
    throw "missing LPBQ sidecars: $sidecarPath"
  }
  $remoteModel = Split-Path -Leaf $modelPath
  $remoteNpuMode = "lpbq_int8"
  $remoteSidecarDir = "$AppDir/sidecars/lpbq_g16_a8w8"
  $remoteR4FwhtV2SidecarDir = "$AppDir/sidecars/r4_fwht_v2"
  $remoteR4HmxDenseFp16SidecarDir = "$AppDir/sidecars/r4_hmx_dense_fp16_v1"
  $runtimeLabel = "isolated-lpbq-a8w8-nonr4"
  $runtimeFiles = @(
    @{ Local = (Join-Path $DeployDir "run_server.sh"); Remote = "run_server.sh"; Always = $true },
    @{ Local = (Join-Path $DeployDir "llama-server"); Remote = "llama-server"; Always = $true },
    @{ Local = (Join-Path $DeployDir "llama-cli"); Remote = "llama-cli"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml.so"); Remote = "libggml.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-base.so"); Remote = "libggml-base.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-cpu.so"); Remote = "libggml-cpu.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libggml-htp.so"); Remote = "libggml-htp.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libllama.so"); Remote = "libllama.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libhtp_ops.so"); Remote = "libhtp_ops.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "libhtp_ops_skel.so"); Remote = "libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "cdsp\libhtp_ops_skel.so"); Remote = "cdsp/libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $DeployDir "dsp\libhtp_ops_skel.so"); Remote = "dsp/libhtp_ops_skel.so"; Always = $true }
  )
} else {
  $iq4Deploy = Join-Path $RepoRoot "artifacts\llm_inference_modes_v73_qwen15b\runtime_deploy\deploy"
  if (-not (Test-Path -LiteralPath $iq4Deploy)) {
    # The isolated pure-FP16 package may be moved outside the llama-npu repo.
    # Keep the original INT4/INT8 route recoverable by probing the current
    # working directory before failing.
    $cwdCandidate = Join-Path (Get-Location).Path "artifacts\llm_inference_modes_v73_qwen15b\runtime_deploy\deploy"
    if (Test-Path -LiteralPath $cwdCandidate) {
      $RepoRoot = (Get-Location).Path
      $iq4Deploy = $cwdCandidate
    }
  }
  if (-not (Test-Path -LiteralPath $iq4Deploy)) {
    throw "missing original INT4/INT8 deploy dir: $iq4Deploy; pass -RepoRootOverride <llama-npu repo root> if the experiment package was moved"
  }
  $modelPath = Join-Path $iq4Deploy "qwen2.5-1.5b-instruct.iq4_nl+q8_0-hmx.gguf"
  $remoteModel = Split-Path -Leaf $modelPath
  $remoteNpuMode = "baseline"
  $runtimeLabel = "original-iq4-runtime"
  $runtimeFiles = @(
    @{ Local = (Join-Path $DeployDir "run_server.sh"); Remote = "run_server.sh"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "llama-server"); Remote = "llama-server"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "llama-cli"); Remote = "llama-cli"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libggml.so"); Remote = "libggml.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libggml-base.so"); Remote = "libggml-base.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libggml-cpu.so"); Remote = "libggml-cpu.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libggml-htp.so"); Remote = "libggml-htp.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libllama.so"); Remote = "libllama.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libhtp_ops.so"); Remote = "libhtp_ops.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libhtp_ops_skel.so"); Remote = "libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libhtp_ops_skel.so"); Remote = "cdsp/libhtp_ops_skel.so"; Always = $true },
    @{ Local = (Join-Path $iq4Deploy "libhtp_ops_skel.so"); Remote = "dsp/libhtp_ops_skel.so"; Always = $true }
  )
}

foreach ($file in $runtimeFiles) {
  Push-IfChanged -SerialArgs $SerialArgs -LocalPath $file.Local -RemotePath "$AppDir/$($file.Remote)" -Always:$([bool]$file.Always)
}
Push-IfChanged -SerialArgs $SerialArgs -LocalPath $modelPath -RemotePath "$AppDir/$remoteModel"
if ($Mode -eq "lpbq_int8") {
  $sidecarFiles = @(Get-ChildItem -LiteralPath $sidecarPath -Recurse -File)
  $safeListPath = Join-Path $sidecarPath "lpbq_v6_full_safe_layers.txt"
  $r4SafeListPath = Join-Path $sidecarPath "lpbq_r4_full_u8_safe_layers.txt"
  # LPBQ deploy-v1 safe-list iteration: exclude the tiny allowlist from the
  # heavy sidecar inventory marker so candidate toggles do not trigger a 4GB
  # sidecar re-push. Its content hash is tracked separately below.
  $sidecarInventoryFiles = @($sidecarFiles | Where-Object { $_.FullName -ne $safeListPath -and $_.FullName -ne $r4SafeListPath })
  $sidecarTotalBytes = ($sidecarInventoryFiles | Measure-Object -Property Length -Sum).Sum
  $safeListHash = if (Test-Path -LiteralPath $safeListPath) {
    (Get-FileHash -LiteralPath $safeListPath -Algorithm SHA256).Hash
  } else {
    "none"
  }
  $r4SafeListHash = if (Test-Path -LiteralPath $r4SafeListPath) {
    (Get-FileHash -LiteralPath $r4SafeListPath -Algorithm SHA256).Hash
  } else {
    "none"
  }
  $sidecarModeLabel = if ($LpbqFullV6Weight) { "fullv6_g$LpbqV6FullGroupTiles" } else { "default" }
  $sidecarInventoryPrefix = "$sidecarModeLabel|$($sidecarInventoryFiles.Count)|$sidecarTotalBytes|"
  $sidecarMarker = "${sidecarInventoryPrefix}safe_list_sha256=$safeListHash|r4_safe_list_sha256=$r4SafeListHash"
  $remoteMarkerPath = "$remoteSidecarDir/.push_marker_$sidecarModeLabel.txt"
  $remoteMarker = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "cat '$remoteMarkerPath' 2>/dev/null || true") -AllowFail
  if ($remoteMarker.Trim() -eq $sidecarMarker) {
    Write-Host "Remote LPBQ sidecars already match marker $sidecarModeLabel; skipping sidecar push."
  } elseif ($remoteMarker.Trim().StartsWith($sidecarInventoryPrefix) -and
            (Test-Path -LiteralPath $safeListPath)) {
    # LPBQ deploy-v1 safe-list iteration: changing only the small allowlist
    # should not force a 4GB sidecar re-push. The inventory prefix proves the
    # remote sidecars already match; update the list and marker only.
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "mkdir -p '$remoteSidecarDir'") | Out-Null
    if (Test-Path -LiteralPath $safeListPath) {
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $safeListPath, "$remoteSidecarDir/lpbq_v6_full_safe_layers.txt") | Out-Null
    }
    if (Test-Path -LiteralPath $r4SafeListPath) {
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $r4SafeListPath, "$remoteSidecarDir/lpbq_r4_full_u8_safe_layers.txt") | Out-Null
    }
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "printf '%s' '$sidecarMarker' > '$remoteMarkerPath'") | Out-Null
  } else {
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "rm -rf '$remoteSidecarDir' && mkdir -p '$AppDir/sidecars'") | Out-Null
    Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $sidecarPath, "$AppDir/sidecars/") | Out-Null
    Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "printf '%s' '$sidecarMarker' > '$remoteMarkerPath'") | Out-Null
  }

  if (Test-Path -LiteralPath $r4FwhtV2SidecarPath) {
    $r4FwhtV2Files = @(Get-ChildItem -LiteralPath $r4FwhtV2SidecarPath -Recurse -File)
    $r4FwhtV2TotalBytes = ($r4FwhtV2Files | Measure-Object -Property Length -Sum).Sum
    $r4FwhtV2ManifestPath = Join-Path $r4FwhtV2SidecarPath "manifest.json"
    $r4FwhtV2ManifestHash = if (Test-Path -LiteralPath $r4FwhtV2ManifestPath) {
      (Get-FileHash -LiteralPath $r4FwhtV2ManifestPath -Algorithm SHA256).Hash
    } else {
      "none"
    }
    $r4FwhtV2Marker = "r4_fwht_v2|$($r4FwhtV2Files.Count)|$r4FwhtV2TotalBytes|manifest_sha256=$r4FwhtV2ManifestHash"
    $remoteR4FwhtV2MarkerPath = "$remoteR4FwhtV2SidecarDir/.push_marker.txt"
    $remoteR4FwhtV2Marker = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "cat '$remoteR4FwhtV2MarkerPath' 2>/dev/null || true") -AllowFail
    if ($remoteR4FwhtV2Marker.Trim() -eq $r4FwhtV2Marker) {
      Write-Host "Remote r4_fwht_v2 sidecars already match marker; skipping v2 sidecar push."
    } else {
      # R4-S1 bridge: keep the large legacy lpbq_g16_a8w8 sidecar marker intact,
      # but publish Guide-v2 schema files next to it so the backend can discover
      # ../r4_fwht_v2/manifest.json from LLAMA_NPU_LPBQ_SIDECAR_DIR.
      Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "rm -rf '$remoteR4FwhtV2SidecarDir' && mkdir -p '$AppDir/sidecars'") | Out-Null
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $r4FwhtV2SidecarPath, "$AppDir/sidecars/") | Out-Null
      Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "printf '%s' '$r4FwhtV2Marker' > '$remoteR4FwhtV2MarkerPath'") | Out-Null
    }
  }

  if ($LpbqR4HmxDenseFp16Sidecar) {
    if (-not (Test-Path -LiteralPath $r4HmxDenseFp16SidecarPath)) {
      throw "missing no-quality/performance-first R4 HMX dense FP16 sidecar: $r4HmxDenseFp16SidecarPath"
    }
    $r4HmxDenseFp16Files = @(Get-ChildItem -LiteralPath $r4HmxDenseFp16SidecarPath -Recurse -File)
    $r4HmxDenseFp16TotalBytes = ($r4HmxDenseFp16Files | Measure-Object -Property Length -Sum).Sum
    $r4HmxDenseFp16ManifestPath = Join-Path $r4HmxDenseFp16SidecarPath "manifest.json"
    $r4HmxDenseFp16ManifestHash = if (Test-Path -LiteralPath $r4HmxDenseFp16ManifestPath) {
      (Get-FileHash -LiteralPath $r4HmxDenseFp16ManifestPath -Algorithm SHA256).Hash
    } else {
      "none"
    }
    $r4HmxDenseFp16Marker = "r4_hmx_dense_fp16_v1|$($r4HmxDenseFp16Files.Count)|$r4HmxDenseFp16TotalBytes|manifest_sha256=$r4HmxDenseFp16ManifestHash"
    $remoteR4HmxDenseFp16MarkerPath = "$remoteR4HmxDenseFp16SidecarDir/.push_marker.txt"
    $remoteR4HmxDenseFp16Marker = Invoke-AdbCapture -SerialArgs $SerialArgs -Args @("shell", "cat '$remoteR4HmxDenseFp16MarkerPath' 2>/dev/null || true") -AllowFail
    if ($remoteR4HmxDenseFp16Marker.Trim() -eq $r4HmxDenseFp16Marker) {
      Write-Host "Remote r4_hmx_dense_fp16_v1 sidecars already match marker; skipping HMX dense sidecar push."
    } else {
      # no-quality/performance-first Stage-A: publish the prepacked HMX dense
      # R4 tiles as a sibling of lpbq_g16_a8w8, matching the host loader.
      Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "rm -rf '$remoteR4HmxDenseFp16SidecarDir' && mkdir -p '$AppDir/sidecars'") | Out-Null
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $r4HmxDenseFp16SidecarPath, "$AppDir/sidecars/") | Out-Null
      Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "printf '%s' '$r4HmxDenseFp16Marker' > '$remoteR4HmxDenseFp16MarkerPath'") | Out-Null
    }
  }

  $remoteNonR4AllowList = ""
  if (-not [string]::IsNullOrWhiteSpace($LpbqFullV6NonR4AllowList)) {
    if (Test-Path -LiteralPath $LpbqFullV6NonR4AllowList) {
      $resolvedNonR4AllowList = (Resolve-Path -LiteralPath $LpbqFullV6NonR4AllowList).Path
      $remoteNonR4AllowList = "$remoteSidecarDir/lpbq_v6_full_non_r4_allow_layers.txt"
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $resolvedNonR4AllowList, $remoteNonR4AllowList) | Out-Null
    } else {
      # Allow advanced runs to point at a pre-existing on-device allowlist.
      $remoteNonR4AllowList = $LpbqFullV6NonR4AllowList
    }
  }

  $remoteR4FullU8SafeList = ""
  if (-not [string]::IsNullOrWhiteSpace($R4FullU8SafeList)) {
    if (Test-Path -LiteralPath $R4FullU8SafeList) {
      $resolvedR4FullU8SafeList = (Resolve-Path -LiteralPath $R4FullU8SafeList).Path
      $remoteR4FullU8SafeList = "$remoteSidecarDir/lpbq_r4_full_u8_safe_layers_override.txt"
      Invoke-Adb -SerialArgs $SerialArgs -Args @("push", $resolvedR4FullU8SafeList, $remoteR4FullU8SafeList) | Out-Null
    } else {
      # Allow advanced runs to point at a pre-existing on-device safe-list.
      $remoteR4FullU8SafeList = $R4FullU8SafeList
    }
  }
}

Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "chmod 755 $AppDir/run_server.sh $AppDir/llama-server $AppDir/llama-cli 2>/dev/null || chmod 755 $AppDir/run_server.sh $AppDir/llama-server") | Out-Null
Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ./run_server.sh stop") -AllowFail | Out-Null
# Keep runtime shape as explicit positional args. The runner still accepts
# LLAMA_N_* env vars as a fallback, but switch/deploy should be reproducible
# from the command line without hidden environment state.
$sidecarPrefix = if ($Mode -eq "lpbq_int8") { "LLAMA_NPU_LPBQ_SIDECAR_DIR='$remoteSidecarDir' " } else { "" }
$enableR4PathPrefix = if ($Mode -eq "lpbq_int8" -and $EnableR4) {
  "LLAMA_NPU_LPBQ_ENABLE_R4_PATH=1 "
} else {
  ""
}
$forceR4Prefix = if ($Mode -eq "lpbq_int8" -and $EnableR4 -and $ForceR4FullU8Safe) {
  $r4IgnorePrefix = if ($ForceR4FullU8IgnoreList) { "LLAMA_NPU_LPBQ_R4_FULL_U8_SAFE_IGNORE_LIST=1 " } else { "" }
  $r4SafeListPrefix = if (-not [string]::IsNullOrWhiteSpace($remoteR4FullU8SafeList)) { "LLAMA_NPU_LPBQ_R4_FULL_U8_SAFE_LIST='$remoteR4FullU8SafeList' " } else { "" }
  "LLAMA_NPU_LPBQ_FORCE_R4_FULL_U8_SAFE=1 ${r4IgnorePrefix}${r4SafeListPrefix}"
} else {
  ""
}
# 2026-07-03 R4 cancellation: R4/FWHT is hard-cancelled for this
# performance track. The old ForceR4FullU8Safe-only behavior is intentionally
# kept as a rollback note in comments above; the -EnableR4 guard near the top
# rejects these runtime R4 env prefixes before deployment can start.
$r4CompactFullU8Prefix = if ($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4CompactFullU8SafeAB) {
  "LLAMA_NPU_LPBQ_R4_COMPACT_FULL_U8_SAFE_AB=1 "
} else {
  ""
}
$fullV6Prefix = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) {
  $nonR4AllowPrefix = if (-not [string]::IsNullOrWhiteSpace($remoteNonR4AllowList)) { "LLAMA_NPU_LPBQ_V6_FULL_NON_R4_ALLOW_LIST='$remoteNonR4AllowList' " } else { "" }
  $nonR4Prefix = if ($LpbqFullV6NonR4) { "LLAMA_NPU_LPBQ_V6_FULL_NON_R4=1 ${nonR4AllowPrefix}" } else { "" }
  # LPBQ 2026-07-03: -LpbqFullV6Weight may still stage non-R4 full-V6 A/Bs,
  # but R4 defaults to compact K-major unless this explicit rollback switch is
  # set. The old implicit R4 full-V6 fd path repeatedly republished weight.
  $r4FullV6FdPrefix = if ($EnableR4 -and $LpbqR4UseFullV6WeightFd) { "LLAMA_NPU_LPBQ_R4_USE_FULL_V6_WEIGHT_FD=1 " } else { "" }
  "LLAMA_NPU_LPBQ_ENABLE_V6_FULL_WEIGHT=1 LLAMA_NPU_LPBQ_V6_FULL_GROUP_TILES='$LpbqV6FullGroupTiles' LLAMA_NPU_LPBQ_R4_FULL_V6_MIN_M='$LpbqR4FullV6MinM' LLAMA_NPU_LPBQ_R4_FULL_V6_SMALL_M_MAX='$LpbqR4FullV6SmallMMax' ${r4FullV6FdPrefix}${nonR4Prefix}"
} else {
  ""
}
# Guide-v2 W0: make the structured-FWHT small-M boundary reproducible in the
# switch manifest. Default remains 4; use -1 only for an explicit all-M probe.
$structuredFwhtPrefix = if ($Mode -eq "lpbq_int8") { "LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX='$LpbqR4StructuredFwhtSmallMMax' " } else { "" }
# Stage-A HMX Dense R4 sidecar is performance-first/no-quality only; the env
# flag is required so default D30 and ordinary LPBQ runs never consume it.
$r4HmxDenseFp16Prefix = if ($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4HmxDenseFp16Sidecar) { "LLAMA_NPU_LPBQ_R4_HMX_DENSE_FP16_SIDECAR=1 " } else { "" }
Invoke-Adb -SerialArgs $SerialArgs -Args @("shell", "cd $AppDir && ${sidecarPrefix}${enableR4PathPrefix}${forceR4Prefix}${r4CompactFullU8Prefix}${fullV6Prefix}${structuredFwhtPrefix}${r4HmxDenseFp16Prefix}./run_server.sh start '$remoteModel' '$remoteNpuMode' '$TraceFlag' '$DetailedFlag' '$Port' '$Batch' '$Ubatch' '$Threads' '$Context' '$Predict' '$NoWarmupFlag'") | Out-Null
Invoke-Adb -SerialArgs $SerialArgs -Args @("forward", "tcp:$Port", "tcp:$Port") | Out-Null
$health = Wait-ServerHealth -Port $Port -TimeoutSec 180

$manifest = [ordered]@{
  switched_at = (Get-Date).ToString("o")
  mode = $Mode
  runtime = $runtimeLabel
  dsp_arch = $DspArch
  remote_app_dir = $AppDir
  port = $Port
  batch = $Batch
  ubatch = $Ubatch
  threads = $Threads
  context = $Context
  predict = $Predict
  trace = [bool]($Trace -or $DetailedTrace)
  detailed_trace = [bool]$DetailedTrace
  no_warmup = [bool]$NoWarmup
  remote_model = $remoteModel
  llama_npu_mode = $remoteNpuMode
  lpbq_sidecar_dir = if ($Mode -eq "lpbq_int8") { $remoteSidecarDir } else { $null }
  lpbq_full_v6_weight = [bool]($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight)
  lpbq_full_v6_non_r4 = [bool]($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight -and $LpbqFullV6NonR4)
  lpbq_full_v6_non_r4_allow_list = if (-not [string]::IsNullOrWhiteSpace($remoteNonR4AllowList)) { $remoteNonR4AllowList } else { $null }
  lpbq_v6_full_group_tiles = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqV6FullGroupTiles } else { $null }
  lpbq_enable_r4_path = [bool]($Mode -eq "lpbq_int8" -and $EnableR4)
  lpbq_r4_use_full_v6_weight_fd = [bool]($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqFullV6Weight -and $LpbqR4UseFullV6WeightFd)
  lpbq_r4_compact_full_u8_safe_ab = [bool]($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4CompactFullU8SafeAB)
  lpbq_force_r4_full_u8_safe = [bool]($Mode -eq "lpbq_int8" -and $EnableR4 -and $ForceR4FullU8Safe)
  lpbq_force_r4_full_u8_ignore_list = [bool]($Mode -eq "lpbq_int8" -and $EnableR4 -and $ForceR4FullU8Safe -and $ForceR4FullU8IgnoreList)
  lpbq_r4_full_u8_safe_list = if (-not [string]::IsNullOrWhiteSpace($remoteR4FullU8SafeList)) { $remoteR4FullU8SafeList } else { $null }
  lpbq_r4_full_v6_min_m = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqR4FullV6MinM } else { $null }
  lpbq_r4_full_v6_small_m_max = if ($Mode -eq "lpbq_int8" -and $LpbqFullV6Weight) { $LpbqR4FullV6SmallMMax } else { $null }
  lpbq_r4_structured_fwht_small_m_max = if ($Mode -eq "lpbq_int8") { $LpbqR4StructuredFwhtSmallMMax } else { $null }
  lpbq_r4_hmx_dense_fp16_sidecar = [bool]($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4HmxDenseFp16Sidecar)
  lpbq_r4_hmx_dense_fp16_sidecar_dir = if ($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4HmxDenseFp16Sidecar) { $remoteR4HmxDenseFp16SidecarDir } else { $null }
  lpbq_r4_hmx_dense_fp16_quality_policy = if ($Mode -eq "lpbq_int8" -and $EnableR4 -and $LpbqR4HmxDenseFp16Sidecar) { "no-quality/performance-first experiment; do not claim deploy-quality" } else { $null }
  health = $health
}
Write-JsonFile -Object $manifest -Path (Join-Path $DeployDir "last_switch_manifest.json")
Write-Host "Switched to $Mode with $remoteModel (LLAMA_NPU_MODE=$remoteNpuMode trace=$TraceFlag detailed=$DetailedFlag no_warmup=$NoWarmupFlag runtime=$runtimeLabel r4_structured_fwht_small_m_max=$($manifest.lpbq_r4_structured_fwht_small_m_max))."
