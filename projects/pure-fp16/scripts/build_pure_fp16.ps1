param(
  [ValidateSet("v73", "v75", "v79")]
  [string] $DspArch = "v73",

  [int] $Jobs = 8,
  [string] $WslDistro = "Ubuntu-22.04",
  [string] $WslEnvScript = "/root/llama-npu-env.sh",
  [string] $AndroidNdkRootWsl = "/root/Qualcomm/Hexagon_SDK/6.3.0.0/tools/android-ndk-r25c",
  [string] $ModelPath = "",
  [switch] $SkipHtpOps,
  [switch] $SkipLlama,
  [switch] $SkipPackage,
  [switch] $ForceConfigure,

  [ValidateSet("hvx", "dma_direct", "dma_scratch_hvx")]
  [string] $Fp16WeightLoadMode = "dma_scratch_hvx",

  [int] $MatmulWeightKb = 1024,
  [int] $MatmulActivationKb = 2048,
  [int] $MatmulOutputKb = 1536,
  [int] $MatmulScratchKb = 1024,
  [int] $MatmulPipelineOutputKb = 512,
  [ValidateSet(0, 1)]
  [int] $Fp16DmaDstBypass = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16DirectFinalDma = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16DirectFinalPipeline = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16DirectFinalCacheInvalidate = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16DirectFinalTouch = 1,
  [ValidateSet(0, 1)]
  [int] $Fp16DecodeDirectPipeline = 0,
  [int] $Fp16DecodePipelineMaxM = 1,
  [ValidateSet(0, 1)]
  [int] $Fp16ParallelWeightPublish = 1,
  [ValidateSet(0, 1)]
  [int] $Fp16PublishUseMemcpy = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16ParallelOutputStore = 1,
  [ValidateSet(0, 1)]
  [int] $Fp16SmallMVtcmEnable = 0,
  [int] $Fp16SmallMThreshold = 192,
  [int] $Fp16SmallMWeightKb = 2048,
  [int] $Fp16SmallMActivationKb = 512,
  [int] $Fp16SmallMOutputKb = 768,
  [int] $Fp16SmallMScratchKb = 2048,
  [int] $Fp16SmallMPipelineOutputKb = 384,
  [int] $Fp16OsMBlock = 512,
  [int] $Fp16OsNBlock = 1024,
  [int] $Fp16OsKBlock = 1024,
  [ValidateSet(0, 1)]
  [int] $Fp16OsWeightGatherDma = 0,
  [ValidateSet(0, 1)]
  [int] $Fp16OsWeightGatherDmaScratch = 1,
  [ValidateSet(0, 1)]
  [int] $Fp16OsWeightGatherDmaPrefillOnly = 1,
  [ValidateSet(0, 1)]
  [int] $MatmulPipelineMode = 0,

  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8Singlepass = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8StreamV6 = 0,
  [int] $LpbqFullU8StreamV6GroupTiles = 16,
  [int] $LpbqR4FullU8StreamV6GroupTiles = 4,
  [int] $LpbqR4FullU8StreamV6SmallMMax = 4,
  [int] $LpbqR4FullU8StreamV6MinM = 0,
  [int] $LpbqR4FullU8StreamV6GroupRoundValue = -1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8StreamV6GroupRoundAlt = 0,
  [int] $LpbqR4FullU8StreamV6GroupRoundAltExtra = 0,
  [int] $LpbqR4FullU8SinglepassRecoverRoundValue = -1,
  [int] $LpbqR4FullU8SinglepassRecoverRoundValueRowsGt1 = -2,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8SinglepassRecoverApply257 = 1,
  [int] $LpbqR4FullU8SinglepassRecoverNegExtra = 0,
  [int] $LpbqR4FullU8SinglepassRecoverCorrRshift = 0,
  [int] $LpbqR4FullU8SinglepassRecoverCorrRshiftRowsGt1 = -1,
  [int] $LpbqR4FullU8SinglepassRecoverCorrExtraRshift = 0,
  [int] $LpbqR4FullU8SinglepassRecoverCorrExtraRshiftRowsGt1 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8StreamV6GroupSignCompensate = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8StreamV6GroupExactRecover = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupExactRecover = 0,
  [ValidateSet(0, 1, 2, 3)]
  [int] $LpbqR4FullU8StreamV6GroupAdaptiveScale = 0,
  [int] $LpbqR4FullU8StreamV6GroupAdaptiveRawAdjust = 0,
  [string] $LpbqR4FullU8StreamV6GroupAdaptiveRoundBiasBits = "0x3800u",
  [string] $LpbqR4FullU8StreamV6GroupScaleBitsOverride = "0",
  [int] $LpbqR4FullU8StreamV6GroupRecoverShiftOverride = -1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8K16Groups = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FullU8K32Groups = 0,
  [string] $LpbqR4FullU8K32ScaleBitsOverride = "0",
  [int] $LpbqR4FullU8K32RecoverShiftOverride = -1,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8PrecompensateHmxScale = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupDelayRecover = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupCompactWeight = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupNTileOverlap = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupNTileOverlapSmallMOnly = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupNTileOverlapPrefillOnly = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8StreamV6GroupNTileOverlapAccFinalIssue = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8V6Copy1024Loop8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8V6CopyBulkLoop8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqSkipUnusedFullU8HiActClear = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHvxWeightCopyPrefetch = 1,
  [ValidateSet(1024, 2048, 4096, 8192, 16384, 32768, 65536)]
  [int] $LpbqHvxWeightCopyPrefetchBytes = 4096,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightGroupCopy = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaLoad = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaDstBypass = 1,
  [ValidateSet(0, 1)]
  # LPBQ K-major W4 DMA reads cached packed weights; default source-bypass off
  # matches the 2026-07-03 standalone/real-layer passing overlap A/B.
  [int] $LpbqKmajorWeightDmaSrcBypass = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaPreissueWait = 1,
  [ValidateSet(0, 1)]
  # Keep the old post-DMA invalidate available, but default it off for the
  # validated K-major W4 overlap path where dst_bypass plus barrier was enough.
  [int] $LpbqKmajorWeightDmaVisibilitySync = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaOverlapAct = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaDoubleBuffer = 0,
  [ValidateRange(0, 512)]
  [int] $LpbqKmajorWeightDmaNBlockTiles = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqKmajorWeightDmaOverlapRowblock = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFfnDownDecodeG32StagingAb = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFfnGateUpDecodeG32StagingAb = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqWeightPublishTraceSplit = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqWeightPathTraceSplit = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqTraceW0ExpertBuckets = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FwhtV2 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4FwhtDirectV6Store = 0,
  [ValidateSet(128)]
  [int] $LpbqR4FwhtBlock = 128,
  [ValidateSet(1, 16, 32)]
  [int] $LpbqR4FwhtScaleGranularity = 16,
  [ValidateSet(0, 1)]
  [int] $LpbqActRows1DirectV6 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActPacketReuse = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActStaticScaleNonR4 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxConvertAffineBench = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxReuseScalePayloadPerNTile = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxHoldAcquirePerRowblock = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxBatchIssueKgroups = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxDoubleBufferDrain = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxEpilogueLagOneTile = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxOpcodeCorpus = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxAfterIssueHvxProbe = 0,
  [ValidateRange(1, 256)]
  [int] $LpbqHmxAfterIssueHvxProbeIters = 8,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightDmaLoad = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightDmaScratch = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightDmaDstBypass = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightDmaOverlapAct = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightDirectMxmemLoad = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightPublishPingpong = 0,
  [ValidateRange(0, 512)]
  [int] $LpbqFullV6WeightPublishPingpongSmallMMax = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightParallelPublish = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6WeightAsyncPrepublishAct = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullV6MacroKgroupPipeline = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActivationHmxCache = 0,
  [int] $LpbqActivationHmxCacheMaxKb = 1536,
  [ValidateSet(0, 1)]
  [int] $LpbqActivationHvxQuant = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActivationFp16StagedQuant = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqDisableOnlineInputScale = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActivationHvxVectorRoundStore = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqActivationHvxFloorRoundStore = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4ParallelPrefill = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4CachedX4KkOuter = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4HvxDot = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4HvxDot16 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4Dot4Rowpair = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4Fp16HmxRotate = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4PrefillHmxDense = 0,
  [int] $LpbqR4PrefillHmxMinM = 32,
  [ValidateSet(128)]
  [int] $LpbqR4HmxDenseBlock = 128,
  [ValidateSet(0, 1)]
  [int] $LpbqR4HmxHighbyteDirectV6Probe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4HmxHfscaleDirectV6Probe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4HmxHfscaleVtcmWeightCacheProbe = 0,
  [ValidateSet(1, 2, 4, 8)]
  [int] $LpbqR4HmxHfscaleVtcmWeightCacheGroups = 8,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128Bulk = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128ParallelRows = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128SkipInnerClear = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128BulkSmallM = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128BulkRowpair = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4SignFhtFast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtForce = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHmxProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHmxIntegrated = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxIntegrated = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxNibbleProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxNibbleIntegrated = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtCoreProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtRowpairDirect = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtI16Direct = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxFullProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxFullIntegrated = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtFp16Direct = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxFullNibbleProbe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StandardFhtHvxFullNibbleIntegrated = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4SignFhtApprox = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4SignFhtColScaleApprox = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4PrefetchCols8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4PrefetchK128RowparCols8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4QuantStore8Fast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4Rows1Reduce8Fast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4Rows1Dot4Split = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4Dot8R128Unroll = 0,
  [ValidateSet(0, 1, 2)]
  [int] $LpbqR4DotPackPrecomputeRowStore = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4K128DotPackPrecomputeRowStore = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128Rowgroup4Dot2 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128ColgroupParallel = 0,
  [int] $LpbqR4V6K128WorkersMax = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128SkipAccZero = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairR128Unroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairDot8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairReduce8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairStore2Fast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairFusedStore4 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairFusedStore4R128Unroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairAlignedLoads = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairPrebaseStore8 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4V6K128RowpairConstLanes = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4ScaleCacheR128Unroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4TraceDotPackSplit = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4NibbleIssueAccOverlap = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4NibbleActiveRowAccum = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4NibbleRawAccum = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4NibbleRawAccumOverlap = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4AccumDecodeUnroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StructuredFwht = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StructuredFwhtStageUnroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4StructuredFwhtDirectK128 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveBucketFast03 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveBucketFast03Rows1 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveAccumVarshiftRows1 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveShift0FastPath = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveUniformShift0NoShift = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveShift0Singlepass = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveProbeScaleCache = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveProbeOnlyShift5Fast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveProbeScalarSmallRows = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveRetrySkipInit = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveScaleLow32Only = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveScaleSkipUnchanged = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveDeriveI32Max = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveScaleWordLut = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqAdaptiveDeriveRow0ExactRows1 = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4GroupedV6RowblockWeightReuse = 0,
  [ValidateSet(1, 2, 4, 8)]
  [int] $LpbqR4GroupedV6RowblockWeightReuseBlocks = 2,
  [ValidateSet(0, 1)]
  [int] $LpbqNonR4GroupedV6RowblockWeightReuse = 0,
  [ValidateSet(1, 2, 4, 8)]
  [int] $LpbqNonR4GroupedV6RowblockWeightReuseBlocks = 2,
  [ValidateSet(0, 1)]
  [int] $LpbqNonR4GroupedV6NBlockRowblockReuse = 0,
  [ValidateRange(1, 128)]
  [int] $LpbqNonR4GroupedV6NBlockRowblockReuseTiles = 8,
  [ValidateSet(0, 1)]
  [int] $LpbqR4GroupedV6RowblockHmxAcquirePerKgroup = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4RowblockProbeOnlyShift5Fast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqR4RowblockSkipInitialScaleLoad = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxFineSubtrace = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8Rows1AccumPairUnroll = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqScaledK128Groups = 0,
  [int] $LpbqScaledKGroupTiles = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqR4ScaledKGroups = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64SafeGroups = 0,
  [ValidateSet(2, 4)]
  [int] $LpbqExactKSafeGroupTiles = 2,
  [ValidateSet(0, 1)]
  [int] $LpbqR4ExactK64SafeGroups = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64NonR4Groups = 0,
  # LPBQ deploy-v1 exact fallback correctness: old default 2184 was 32767/15;
  # exact q4<<1 drains require 32767/30.
  [int] $LpbqExactK64SafeAbsLimit = 1092,
  [int] $LpbqExactK64DebugStopStage = 0,
  [ValidateSet(0, 1, 2, 3, 4, 5, 6, 7)]
  [int] $LpbqMode166StopAfter = 0,
  [ValidateSet(0, 1, 2, 3)]
  [int] $LpbqMode166V6OneStopStage = 0,
  [int] $LpbqExactK64MinSafePercent = 50,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64AcquirePerGroup = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64IssueAccOverlap = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64FullWeightStream = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64PairPreaccum = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64RequireAllSafe = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqExactK64ForceUnsafe = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxClearOutBeforeStore = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxClearActiveRowsOnly = 1,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxCvtUhDrain = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxCvtUhSelector2Drain = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqFullU8AcquirePerM = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqHmxAcquirePerM = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqImmutableSidecarCache = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqRoundhalfFoldedStoreFast = 0,
  [ValidateSet(0, 1)]
  [int] $LpbqRoundhalfFoldedRawAdjustBiasFast = 0
)

. "$PSScriptRoot\common.ps1"

$ExperimentRoot = Get-ExperimentRoot
$HtpRoot = Join-Path $ExperimentRoot "src\htp-ops-lib-main"
$LlamaRoot = Join-Path $ExperimentRoot "src\llama.cpp-npu-htp-backend"
$BuildDirName = "build-android-htp-$DspArch-shared"
$LlamaBuildDir = Join-Path $LlamaRoot $BuildDirName
$DeployDir = Join-Path $ExperimentRoot "deploy"
$DeployTemplate = Join-Path $ExperimentRoot "deploy-template\run_server.sh"
$LogPath = Join-Path $ExperimentRoot "logs\build_pure_fp16_$(Get-Date -Format yyyyMMdd_HHmmss).log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath), $DeployDir | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $DeployDir "run_server.sh"))) {
  if (-not (Test-Path -LiteralPath $DeployTemplate)) {
    throw "missing deploy runner template: $DeployTemplate"
  }
  Copy-Item -LiteralPath $DeployTemplate -Destination (Join-Path $DeployDir "run_server.sh")
}

$Fp16WeightLoadModeValue = switch ($Fp16WeightLoadMode) {
  "hvx" { 0 }
  "dma_direct" { 1 }
  "dma_scratch_hvx" { 2 }
}

function Invoke-WslLogged {
  param([Parameter(Mandatory = $true)][string] $Command)
  Write-Host "[wsl] $Command"
  Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value "`n===== WSL COMMAND =====`n$Command"
  $oldErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & wsl.exe -d $WslDistro -u root -- bash -lc $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
  if ($LASTEXITCODE -ne 0) {
    $ErrorActionPreference = $oldErrorActionPreference
    throw "WSL command failed with exit code $LASTEXITCODE. See $LogPath"
  }
  $ErrorActionPreference = $oldErrorActionPreference
}

if (-not $SkipHtpOps) {
  $htpWsl = ConvertTo-WslPath $HtpRoot
  if ($ForceConfigure) {
    $htpGeneratedDirs = @(
      (Join-Path $HtpRoot "android_ReleaseG_aarch64"),
      (Join-Path $HtpRoot "hexagon_ReleaseG_toolv88_$DspArch")
    )
    foreach ($htpBuildDir in $htpGeneratedDirs) {
      if (-not (Test-Path -LiteralPath $htpBuildDir)) {
        continue
      }
      $resolvedBuild = (Resolve-Path -LiteralPath $htpBuildDir).Path
      $resolvedRoot = (Resolve-Path -LiteralPath $HtpRoot).Path
      if (-not $resolvedBuild.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to clean generated HTP CMake cache outside isolated htp tree: $resolvedBuild"
      }
      # LPBQ macro-pipeline A/B note: HTP build_cmake can keep old knob values
      # in generated CMake state. The old metadata-only cleanup is kept below as
      # a rollback breadcrumb, but it can leave stale build.ninja without the
      # CMakeFiles target dirs and break depfile creation after -ForceConfigure.
      # Remove-Item -LiteralPath (Join-Path $htpBuildDir "CMakeCache.txt") -Force -ErrorAction SilentlyContinue
      # Remove-Item -LiteralPath (Join-Path $htpBuildDir "CMakeFiles") -Recurse -Force -ErrorAction SilentlyContinue
      Remove-Item -LiteralPath $resolvedBuild -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
  $matmulDefinePairs = @(
    "HTP_FP16_WEIGHT_LOAD_MODE=$Fp16WeightLoadModeValue",
    "HTP_MATMUL_PIPELINE_MODE=$MatmulPipelineMode",
    "HTP_MATMUL_WEIGHT_AREA_KB=$MatmulWeightKb",
    "HTP_MATMUL_ACTIVATION_AREA_KB=$MatmulActivationKb",
    "HTP_MATMUL_OUTPUT_AREA_KB=$MatmulOutputKb",
    "HTP_MATMUL_SCRATCH_AREA_KB=$MatmulScratchKb",
    "HTP_MATMUL_PIPELINE_OUTPUT_KB=$MatmulPipelineOutputKb",
    "HTP_FP16_DMA_DST_BYPASS=$Fp16DmaDstBypass",
    "HTP_FP16_DIRECT_FINAL_DMA=$Fp16DirectFinalDma",
    "HTP_FP16_DIRECT_FINAL_PIPELINE=$Fp16DirectFinalPipeline",
    "HTP_FP16_DIRECT_FINAL_CACHE_INVALIDATE=$Fp16DirectFinalCacheInvalidate",
    "HTP_FP16_DIRECT_FINAL_TOUCH=$Fp16DirectFinalTouch",
    "HTP_FP16_DECODE_DIRECT_PIPELINE=$Fp16DecodeDirectPipeline",
    "HTP_FP16_DECODE_PIPELINE_MAX_M=$Fp16DecodePipelineMaxM",
    "HTP_FP16_PARALLEL_WEIGHT_PUBLISH=$Fp16ParallelWeightPublish",
    "HTP_FP16_PUBLISH_USE_MEMCPY=$Fp16PublishUseMemcpy",
    "HTP_FP16_PARALLEL_OUTPUT_STORE=$Fp16ParallelOutputStore",
    "HTP_FP16_SMALL_M_VTCM_ENABLE=$Fp16SmallMVtcmEnable",
    "HTP_FP16_SMALL_M_THRESHOLD=$Fp16SmallMThreshold",
    "HTP_FP16_SMALL_M_WEIGHT_AREA_KB=$Fp16SmallMWeightKb",
    "HTP_FP16_SMALL_M_ACTIVATION_AREA_KB=$Fp16SmallMActivationKb",
    "HTP_FP16_SMALL_M_OUTPUT_AREA_KB=$Fp16SmallMOutputKb",
    "HTP_FP16_SMALL_M_SCRATCH_AREA_KB=$Fp16SmallMScratchKb",
    "HTP_FP16_SMALL_M_PIPELINE_OUTPUT_KB=$Fp16SmallMPipelineOutputKb",
    "HTP_FP16_OS_M_BLOCK_SIZE=$Fp16OsMBlock",
    "HTP_FP16_OS_N_BLOCK_SIZE=$Fp16OsNBlock",
    "HTP_FP16_OS_K_BLOCK_SIZE=$Fp16OsKBlock",
    "HTP_FP16_OS_WEIGHT_GATHER_DMA=$Fp16OsWeightGatherDma",
    "HTP_FP16_OS_WEIGHT_GATHER_DMA_SCRATCH=$Fp16OsWeightGatherDmaScratch",
    "HTP_FP16_OS_WEIGHT_GATHER_DMA_PREFILL_ONLY=$Fp16OsWeightGatherDmaPrefillOnly",
    # LPBQ deploy-v1 R4 A/B knobs. Defaults keep the restored correct baseline;
    # explicit nonzero values are for standalone-first FP16-style A8W8 probes.
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS=$LpbqR4FullU8Singlepass",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6=$LpbqR4FullU8StreamV6",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_TILES=$LpbqFullU8StreamV6GroupTiles",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_TILES=$LpbqR4FullU8StreamV6GroupTiles",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_SMALL_M_MAX=$LpbqR4FullU8StreamV6SmallMMax",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_MIN_M=$LpbqR4FullU8StreamV6MinM",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_VALUE=$LpbqR4FullU8StreamV6GroupRoundValue",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT=$LpbqR4FullU8StreamV6GroupRoundAlt",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT_EXTRA=$LpbqR4FullU8StreamV6GroupRoundAltExtra",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE=$LpbqR4FullU8SinglepassRecoverRoundValue",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE_ROWS_GT1=$LpbqR4FullU8SinglepassRecoverRoundValueRowsGt1",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_APPLY_257=$LpbqR4FullU8SinglepassRecoverApply257",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_NEG_EXTRA=$LpbqR4FullU8SinglepassRecoverNegExtra",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT=$LpbqR4FullU8SinglepassRecoverCorrRshift",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT_ROWS_GT1=$LpbqR4FullU8SinglepassRecoverCorrRshiftRowsGt1",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT=$LpbqR4FullU8SinglepassRecoverCorrExtraRshift",
    "HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT_ROWS_GT1=$LpbqR4FullU8SinglepassRecoverCorrExtraRshiftRowsGt1",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SIGN_COMPENSATE=$LpbqR4FullU8StreamV6GroupSignCompensate",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER=$LpbqR4FullU8StreamV6GroupExactRecover",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER=$LpbqFullU8StreamV6GroupExactRecover",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_SCALE=$LpbqR4FullU8StreamV6GroupAdaptiveScale",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_RAW_ADJUST=$LpbqR4FullU8StreamV6GroupAdaptiveRawAdjust",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_ROUND_BIAS_BITS=$LpbqR4FullU8StreamV6GroupAdaptiveRoundBiasBits",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SCALE_BITS_OVERRIDE=$LpbqR4FullU8StreamV6GroupScaleBitsOverride",
    "HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_RECOVER_SHIFT_OVERRIDE=$LpbqR4FullU8StreamV6GroupRecoverShiftOverride",
    "HTP_LPBQ_R4_FULL_U8_K16_GROUPS=$LpbqR4FullU8K16Groups",
    "HTP_LPBQ_R4_FULL_U8_K32_GROUPS=$LpbqR4FullU8K32Groups",
    "HTP_LPBQ_R4_FULL_U8_K32_SCALE_BITS_OVERRIDE=$LpbqR4FullU8K32ScaleBitsOverride",
    "HTP_LPBQ_R4_FULL_U8_K32_RECOVER_SHIFT_OVERRIDE=$LpbqR4FullU8K32RecoverShiftOverride",
    "HTP_LPBQ_FULL_U8_PRECOMPENSATE_HMX_SCALE=$LpbqFullU8PrecompensateHmxScale",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_DELAY_RECOVER=$LpbqFullU8StreamV6GroupDelayRecover",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_COMPACT_WEIGHT=$LpbqFullU8StreamV6GroupCompactWeight",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP=$LpbqFullU8StreamV6GroupNTileOverlap",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_SMALL_M_ONLY=$LpbqFullU8StreamV6GroupNTileOverlapSmallMOnly",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_PREFILL_ONLY=$LpbqFullU8StreamV6GroupNTileOverlapPrefillOnly",
    "HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_ACC_FINAL_ISSUE=$LpbqFullU8StreamV6GroupNTileOverlapAccFinalIssue",
    "HTP_LPBQ_FULL_U8_V6_COPY_1024_LOOP8=$LpbqFullU8V6Copy1024Loop8",
    "HTP_LPBQ_FULL_U8_V6_COPY_BULK_LOOP8=$LpbqFullU8V6CopyBulkLoop8",
    "HTP_LPBQ_SKIP_UNUSED_FULL_U8_HI_ACT_CLEAR=$LpbqSkipUnusedFullU8HiActClear",
    "HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH=$LpbqHvxWeightCopyPrefetch",
    "HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH_BYTES=$LpbqHvxWeightCopyPrefetchBytes",
    "HTP_LPBQ_KMAJOR_WEIGHT_GROUP_COPY=$LpbqKmajorWeightGroupCopy",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_LOAD=$LpbqKmajorWeightDmaLoad",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_DST_BYPASS=$LpbqKmajorWeightDmaDstBypass",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_SRC_BYPASS=$LpbqKmajorWeightDmaSrcBypass",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_PREISSUE_WAIT=$LpbqKmajorWeightDmaPreissueWait",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_VISIBILITY_SYNC=$LpbqKmajorWeightDmaVisibilitySync",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ACT=$LpbqKmajorWeightDmaOverlapAct",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_DOUBLE_BUFFER=$LpbqKmajorWeightDmaDoubleBuffer",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_N_BLOCK_TILES=$LpbqKmajorWeightDmaNBlockTiles",
    "HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ROWBLOCK=$LpbqKmajorWeightDmaOverlapRowblock",
    "HTP_LPBQ_FFNDOWN_DECODE_G32_STAGING_AB=$LpbqFfnDownDecodeG32StagingAb",
    "HTP_LPBQ_FFNGATEUP_DECODE_G32_STAGING_AB=$LpbqFfnGateUpDecodeG32StagingAb",
    "HTP_LPBQ_WEIGHT_PUBLISH_TRACE_SPLIT=$LpbqWeightPublishTraceSplit",
    "HTP_LPBQ_WEIGHT_PATH_TRACE_SPLIT=$LpbqWeightPathTraceSplit",
    "HTP_LPBQ_TRACE_W0_EXPERT_BUCKETS=$LpbqTraceW0ExpertBuckets",
    "HTP_LPBQ_R4_FWHT_V2=$LpbqR4FwhtV2",
    "HTP_LPBQ_R4_FWHT_DIRECT_V6_STORE=$LpbqR4FwhtDirectV6Store",
    "HTP_LPBQ_R4_FWHT_BLOCK=$LpbqR4FwhtBlock",
    "HTP_LPBQ_R4_FWHT_SCALE_GRANULARITY=$LpbqR4FwhtScaleGranularity",
    "HTP_LPBQ_ACT_ROWS1_DIRECT_V6=$LpbqActRows1DirectV6",
    "HTP_LPBQ_ACT_PACKET_REUSE=$LpbqActPacketReuse",
    "HTP_LPBQ_ACT_STATIC_SCALE_NONR4=$LpbqActStaticScaleNonR4",
    "HTP_LPBQ_HMX_CONVERT_AFFINE_BENCH=$LpbqHmxConvertAffineBench",
    "HTP_LPBQ_HMX_REUSE_SCALE_PAYLOAD_PER_NTILE=$LpbqHmxReuseScalePayloadPerNTile",
    "HTP_LPBQ_HMX_HOLD_ACQUIRE_PER_ROWBLOCK=$LpbqHmxHoldAcquirePerRowblock",
    "HTP_LPBQ_HMX_BATCH_ISSUE_KGROUPS=$LpbqHmxBatchIssueKgroups",
    "HTP_LPBQ_HMX_DOUBLE_BUFFER_DRAIN=$LpbqHmxDoubleBufferDrain",
    "HTP_LPBQ_HMX_EPILOGUE_LAG_ONE_TILE=$LpbqHmxEpilogueLagOneTile",
    "HTP_LPBQ_HMX_OPCODE_CORPUS=$LpbqHmxOpcodeCorpus",
    "HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE=$LpbqHmxAfterIssueHvxProbe",
    "HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE_ITERS=$LpbqHmxAfterIssueHvxProbeIters",
    "HTP_LPBQ_FULL_V6_WEIGHT_DMA_LOAD=$LpbqFullV6WeightDmaLoad",
    "HTP_LPBQ_FULL_V6_WEIGHT_DMA_SCRATCH=$LpbqFullV6WeightDmaScratch",
    "HTP_LPBQ_FULL_V6_WEIGHT_DMA_DST_BYPASS=$LpbqFullV6WeightDmaDstBypass",
    "HTP_LPBQ_FULL_V6_WEIGHT_DMA_OVERLAP_ACT=$LpbqFullV6WeightDmaOverlapAct",
    "HTP_LPBQ_FULL_V6_WEIGHT_DIRECT_MXMEM_LOAD=$LpbqFullV6WeightDirectMxmemLoad",
    "HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG=$LpbqFullV6WeightPublishPingpong",
    "HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG_SMALL_M_MAX=$LpbqFullV6WeightPublishPingpongSmallMMax",
    "HTP_LPBQ_FULL_V6_WEIGHT_PARALLEL_PUBLISH=$LpbqFullV6WeightParallelPublish",
    "HTP_LPBQ_FULL_V6_WEIGHT_ASYNC_PREPUBLISH_ACT=$LpbqFullV6WeightAsyncPrepublishAct",
    "HTP_LPBQ_FULL_V6_MACRO_KGROUP_PIPELINE=$LpbqFullV6MacroKgroupPipeline",
    "HTP_LPBQ_ACTIVATION_HMX_CACHE=$LpbqActivationHmxCache",
    "HTP_LPBQ_ACTIVATION_HMX_CACHE_MAX_KB=$LpbqActivationHmxCacheMaxKb",
    "HTP_LPBQ_ACTIVATION_HVX_QUANT=$LpbqActivationHvxQuant",
    "HTP_LPBQ_ACTIVATION_FP16_STAGED_QUANT=$LpbqActivationFp16StagedQuant",
    "HTP_LPBQ_DISABLE_ONLINE_INPUT_SCALE=$LpbqDisableOnlineInputScale",
    "HTP_LPBQ_ACTIVATION_HVX_VECTOR_ROUND_STORE=$LpbqActivationHvxVectorRoundStore",
    "HTP_LPBQ_ACTIVATION_HVX_FLOOR_ROUND_STORE=$LpbqActivationHvxFloorRoundStore",
    "HTP_LPBQ_R4_PARALLEL_PREFILL=$LpbqR4ParallelPrefill",
    "HTP_LPBQ_R4_CACHED_X4_KK_OUTER=$LpbqR4CachedX4KkOuter",
    "HTP_LPBQ_R4_HVX_DOT=$LpbqR4HvxDot",
    "HTP_LPBQ_R4_HVX_DOT16=$LpbqR4HvxDot16",
    "HTP_LPBQ_R4_DOT4_ROWPAIR=$LpbqR4Dot4Rowpair",
    "HTP_LPBQ_R4_FP16_HMX_ROTATE=$LpbqR4Fp16HmxRotate",
    "HTP_LPBQ_R4_PREFILL_HMX_DENSE=$LpbqR4PrefillHmxDense",
    "HTP_LPBQ_R4_PREFILL_HMX_MIN_M=$LpbqR4PrefillHmxMinM",
    "HTP_LPBQ_R4_HMX_DENSE_BLOCK=$LpbqR4HmxDenseBlock",
    "HTP_LPBQ_R4_HMX_HIGHBYTE_DIRECT_V6_PROBE=$LpbqR4HmxHighbyteDirectV6Probe",
    "HTP_LPBQ_R4_HMX_HFSCALE_DIRECT_V6_PROBE=$LpbqR4HmxHfscaleDirectV6Probe",
    "HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_PROBE=$LpbqR4HmxHfscaleVtcmWeightCacheProbe",
    "HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_GROUPS=$LpbqR4HmxHfscaleVtcmWeightCacheGroups",
    "HTP_LPBQ_R4_V6_K128_BULK=$LpbqR4V6K128Bulk",
    "HTP_LPBQ_R4_V6_K128_PARALLEL_ROWS=$LpbqR4V6K128ParallelRows",
    "HTP_LPBQ_R4_V6_K128_SKIP_INNER_CLEAR=$LpbqR4V6K128SkipInnerClear",
    "HTP_LPBQ_R4_V6_K128_BULK_SMALL_M=$LpbqR4V6K128BulkSmallM",
    # LPBQ deploy-v1 2026-07-02: pass every cache-backed K128 rowpair switch
    # explicitly so a previous no-quality experiment cannot leak through CMakeCache.
    "HTP_LPBQ_R4_V6_K128_BULK_ROWPAIR=$LpbqR4V6K128BulkRowpair",
    "HTP_LPBQ_R4_SIGN_FHT_FAST=$LpbqR4SignFhtFast",
    "HTP_LPBQ_R4_STANDARD_FHT_FORCE=$LpbqR4StandardFhtForce",
    "HTP_LPBQ_R4_STANDARD_FHT_HMX_PROBE=$LpbqR4StandardFhtHmxProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_HMX_INTEGRATED=$LpbqR4StandardFhtHmxIntegrated",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_PROBE=$LpbqR4StandardFhtHvxProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_INTEGRATED=$LpbqR4StandardFhtHvxIntegrated",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_PROBE=$LpbqR4StandardFhtHvxNibbleProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_INTEGRATED=$LpbqR4StandardFhtHvxNibbleIntegrated",
    "HTP_LPBQ_R4_STANDARD_FHT_CORE_PROBE=$LpbqR4StandardFhtCoreProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_ROWPAIR_DIRECT=$LpbqR4StandardFhtRowpairDirect",
    "HTP_LPBQ_R4_STANDARD_FHT_I16_DIRECT=$LpbqR4StandardFhtI16Direct",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_PROBE=$LpbqR4StandardFhtHvxFullProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_INTEGRATED=$LpbqR4StandardFhtHvxFullIntegrated",
    "HTP_LPBQ_R4_STANDARD_FHT_FP16_DIRECT=$LpbqR4StandardFhtFp16Direct",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_PROBE=$LpbqR4StandardFhtHvxFullNibbleProbe",
    "HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_INTEGRATED=$LpbqR4StandardFhtHvxFullNibbleIntegrated",
    "HTP_LPBQ_R4_SIGN_FHT_APPROX=$LpbqR4SignFhtApprox",
    "HTP_LPBQ_R4_SIGN_FHT_COLSCALE_APPROX=$LpbqR4SignFhtColScaleApprox",
    "HTP_LPBQ_R4_PREFETCH_COLS8=$LpbqR4PrefetchCols8",
    "HTP_LPBQ_R4_PREFETCH_K128_ROWPAR_COLS8=$LpbqR4PrefetchK128RowparCols8",
    # LPBQ deploy-v1 2026-06-17: keep the older nibble-store8 probe disabled;
    # this workflow parameter now drives only the narrower grouped-V6 U8 store8 A/B.
    # "HTP_LPBQ_R4_QUANT_STORE8_FAST=$LpbqR4QuantStore8Fast",
    "HTP_LPBQ_R4_U8_V6_QUANT_STORE8_FAST=$LpbqR4QuantStore8Fast",
    "HTP_LPBQ_R4_ROWS1_REDUCE8_FAST=$LpbqR4Rows1Reduce8Fast",
    "HTP_LPBQ_R4_ROWS1_DOT4_SPLIT=$LpbqR4Rows1Dot4Split",
    "HTP_LPBQ_R4_DOT8_R128_UNROLL=$LpbqR4Dot8R128Unroll",
    "HTP_LPBQ_R4_DOT_PACK_PRECOMPUTE_ROW_STORE=$LpbqR4DotPackPrecomputeRowStore",
    "HTP_LPBQ_R4_K128_DOT_PACK_PRECOMPUTE_ROW_STORE=$LpbqR4K128DotPackPrecomputeRowStore",
    "HTP_LPBQ_R4_V6_K128_ROWGROUP4_DOT2=$LpbqR4V6K128Rowgroup4Dot2",
    "HTP_LPBQ_R4_V6_K128_COLGROUP_PARALLEL=$LpbqR4V6K128ColgroupParallel",
    "HTP_LPBQ_R4_V6_K128_WORKERS_MAX=$LpbqR4V6K128WorkersMax",
    "HTP_LPBQ_R4_V6_K128_SKIP_ACC_ZERO=$LpbqR4V6K128SkipAccZero",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_R128_UNROLL=$LpbqR4V6K128RowpairR128Unroll",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_DOT8=$LpbqR4V6K128RowpairDot8",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_REDUCE8=$LpbqR4V6K128RowpairReduce8",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_STORE2_FAST=$LpbqR4V6K128RowpairStore2Fast",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4=$LpbqR4V6K128RowpairFusedStore4",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4_R128_UNROLL=$LpbqR4V6K128RowpairFusedStore4R128Unroll",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_ALIGNED_LOADS=$LpbqR4V6K128RowpairAlignedLoads",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_PREBASE_STORE8=$LpbqR4V6K128RowpairPrebaseStore8",
    "HTP_LPBQ_R4_V6_K128_ROWPAIR_CONST_LANES=$LpbqR4V6K128RowpairConstLanes",
    "HTP_LPBQ_R4_SCALE_CACHE_R128_UNROLL=$LpbqR4ScaleCacheR128Unroll",
    "HTP_LPBQ_R4_TRACE_DOT_PACK_SPLIT=$LpbqR4TraceDotPackSplit",
    "HTP_LPBQ_R4_NIBBLE_ISSUE_ACC_OVERLAP=$LpbqR4NibbleIssueAccOverlap",
    "HTP_LPBQ_R4_NIBBLE_ACTIVE_ROW_ACCUM=$LpbqR4NibbleActiveRowAccum",
    "HTP_LPBQ_R4_NIBBLE_RAW_ACCUM=$LpbqR4NibbleRawAccum",
    "HTP_LPBQ_R4_NIBBLE_RAW_ACCUM_OVERLAP=$LpbqR4NibbleRawAccumOverlap",
    "HTP_LPBQ_R4_ACCUM_DECODE_UNROLL=$LpbqR4AccumDecodeUnroll",
    # LPBQ deploy-v1 2026-07-02: expose the older structured-FWHT sidecar path
    # as a build knob. Default stays enabled to preserve prior behavior; setting
    # this to 0 is only a Stage-A-only D30/source parity probe, not code removal.
    "HTP_LPBQ_R4_STRUCTURED_FWHT=$LpbqR4StructuredFwht",
    "HTP_LPBQ_R4_STRUCTURED_FWHT_STAGE_UNROLL=$LpbqR4StructuredFwhtStageUnroll",
    "HTP_LPBQ_R4_STRUCTURED_FWHT_DIRECT_K128=$LpbqR4StructuredFwhtDirectK128",
    "HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3=$LpbqAdaptiveBucketFast03",
    "HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3_ROWS1=$LpbqAdaptiveBucketFast03Rows1",
    "HTP_LPBQ_ADAPTIVE_ACCUM_VARSHIFT_ROWS1=$LpbqAdaptiveAccumVarshiftRows1",
    "HTP_LPBQ_ADAPTIVE_SHIFT0_FAST_PATH=$LpbqAdaptiveShift0FastPath",
    "HTP_LPBQ_ADAPTIVE_UNIFORM_SHIFT0_NOSHIFT=$LpbqAdaptiveUniformShift0NoShift",
    "HTP_LPBQ_ADAPTIVE_SHIFT0_SINGLEPASS=$LpbqAdaptiveShift0Singlepass",
    "HTP_LPBQ_ADAPTIVE_PROBE_SCALE_CACHE=$LpbqAdaptiveProbeScaleCache",
    "HTP_LPBQ_ADAPTIVE_PROBE_ONLY_SHIFT5_FAST=$LpbqAdaptiveProbeOnlyShift5Fast",
    "HTP_LPBQ_ADAPTIVE_PROBE_SCALAR_SMALL_ROWS=$LpbqAdaptiveProbeScalarSmallRows",
    "HTP_LPBQ_ADAPTIVE_RETRY_SKIP_INIT=$LpbqAdaptiveRetrySkipInit",
    "HTP_LPBQ_ADAPTIVE_SCALE_LOW32_ONLY=$LpbqAdaptiveScaleLow32Only",
    "HTP_LPBQ_ADAPTIVE_SCALE_SKIP_UNCHANGED=$LpbqAdaptiveScaleSkipUnchanged",
    "HTP_LPBQ_ADAPTIVE_DERIVE_I32_MAX=$LpbqAdaptiveDeriveI32Max",
    "HTP_LPBQ_ADAPTIVE_SCALE_WORD_LUT=$LpbqAdaptiveScaleWordLut",
    "HTP_LPBQ_ADAPTIVE_DERIVE_ROW0_EXACT_ROWS1=$LpbqAdaptiveDeriveRow0ExactRows1",
    "HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE=$LpbqR4GroupedV6RowblockWeightReuse",
    "HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS=$LpbqR4GroupedV6RowblockWeightReuseBlocks",
    "HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE=$LpbqNonR4GroupedV6RowblockWeightReuse",
    "HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS=$LpbqNonR4GroupedV6RowblockWeightReuseBlocks",
    "HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE=$LpbqNonR4GroupedV6NBlockRowblockReuse",
    "HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE_TILES=$LpbqNonR4GroupedV6NBlockRowblockReuseTiles",
    "HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_HMX_ACQUIRE_PER_KGROUP=$LpbqR4GroupedV6RowblockHmxAcquirePerKgroup",
    "HTP_LPBQ_R4_ROWBLOCK_PROBE_ONLY_SHIFT5_FAST=$LpbqR4RowblockProbeOnlyShift5Fast",
    "HTP_LPBQ_R4_ROWBLOCK_SKIP_INITIAL_SCALE_LOAD=$LpbqR4RowblockSkipInitialScaleLoad",
    "HTP_LPBQ_HMX_FINE_SUBTRACE=$LpbqHmxFineSubtrace",
    "HTP_LPBQ_FULL_U8_ROWS1_ACCUM_PAIR_UNROLL=$LpbqFullU8Rows1AccumPairUnroll",
    "HTP_LPBQ_SCALED_K128_GROUPS=$LpbqScaledK128Groups",
    "HTP_LPBQ_SCALED_K_GROUP_TILES=$LpbqScaledKGroupTiles",
    "HTP_LPBQ_R4_SCALED_K_GROUPS=$LpbqR4ScaledKGroups",
    "HTP_LPBQ_EXACT_K64_SAFE_GROUPS=$LpbqExactK64SafeGroups",
    "HTP_LPBQ_EXACT_K_SAFE_GROUP_TILES=$LpbqExactKSafeGroupTiles",
    "HTP_LPBQ_R4_EXACT_K64_SAFE_GROUPS=$LpbqR4ExactK64SafeGroups",
    "HTP_LPBQ_EXACT_K64_NON_R4_GROUPS=$LpbqExactK64NonR4Groups",
    "HTP_LPBQ_EXACT_K64_SAFE_ABS_LIMIT=$LpbqExactK64SafeAbsLimit",
    "HTP_LPBQ_EXACT_K64_DEBUG_STOP_STAGE=$LpbqExactK64DebugStopStage",
    "HTP_LPBQ_MODE166_STOP_AFTER=$LpbqMode166StopAfter",
    "HTP_LPBQ_MODE166_V6_ONE_STOP_STAGE=$LpbqMode166V6OneStopStage",
    "HTP_LPBQ_EXACT_K64_MIN_SAFE_PERCENT=$LpbqExactK64MinSafePercent",
    "HTP_LPBQ_EXACT_K64_ACQUIRE_PER_GROUP=$LpbqExactK64AcquirePerGroup",
    "HTP_LPBQ_EXACT_K64_ISSUE_ACC_OVERLAP=$LpbqExactK64IssueAccOverlap",
    "HTP_LPBQ_EXACT_K64_FULL_WEIGHT_STREAM=$LpbqExactK64FullWeightStream",
    "HTP_LPBQ_EXACT_K64_PAIR_PREACCUM=$LpbqExactK64PairPreaccum",
    "HTP_LPBQ_EXACT_K64_REQUIRE_ALL_SAFE=$LpbqExactK64RequireAllSafe",
    "HTP_LPBQ_EXACT_K64_FORCE_UNSAFE=$LpbqExactK64ForceUnsafe",
    # LPBQ deploy-v1 drain A/B knobs. Defaults preserve the current SOTA; use
    # clear-out=0 only behind standalone correctness gates.
    "HTP_LPBQ_HMX_CLEAR_OUT_BEFORE_STORE=$LpbqHmxClearOutBeforeStore",
    "HTP_LPBQ_HMX_CLEAR_ACTIVE_ROWS_ONLY=$LpbqHmxClearActiveRowsOnly",
    "HTP_LPBQ_HMX_CVT_UH_DRAIN=$LpbqHmxCvtUhDrain",
    "HTP_LPBQ_HMX_CVT_UH_SELECTOR2_DRAIN=$LpbqHmxCvtUhSelector2Drain",
    "HTP_LPBQ_FULL_U8_ACQUIRE_PER_M=$LpbqFullU8AcquirePerM",
    "HTP_LPBQ_HMX_ACQUIRE_PER_M=$LpbqHmxAcquirePerM",
    "HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE=$LpbqImmutableSidecarCache",
    "HTP_LPBQ_ROUNDHALF_FOLDED_STORE_FAST=$LpbqRoundhalfFoldedStoreFast",
    "HTP_LPBQ_ROUNDHALF_FOLDED_RAWADJ_BIAS_FAST=$LpbqRoundhalfFoldedRawAdjustBiasFast"
  )
  $matmulDefines = $matmulDefinePairs -join " "
  $matmulCmakeDefines = ($matmulDefinePairs | ForEach-Object { "-D$_" }) -join " "
  $matmulEnvExports = ($matmulDefinePairs | ForEach-Object { "export $_" }) -join "; "
  # LPBQ deploy-v1 2026-07-02 source-D30 parity: the android host cache also
  # stores these macros, so keep it synchronized with the hexagon skel build.
  # Old lines kept for rollback. The bare VAR=VALUE form can be ignored by a
  # fresh HTP build_cmake configure; the -D form is also kept because the SDK
  # wrapper can turn it into cache variables literally named "-D...".
  # $cmd = "set -eo pipefail; source /root/llama-npu-env.sh >/dev/null 2>&1; cd '$htpWsl'; build_cmake android; build_cmake hexagon DSP_ARCH=$DspArch FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF $matmulDefines"
  # $cmd = "set -eo pipefail; source /root/llama-npu-env.sh >/dev/null 2>&1; cd '$htpWsl'; build_cmake android $matmulDefines; build_cmake hexagon DSP_ARCH=$DspArch FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF $matmulDefines"
  # $cmd = "set -eo pipefail; source /root/llama-npu-env.sh >/dev/null 2>&1; cd '$htpWsl'; build_cmake android $matmulCmakeDefines; build_cmake hexagon DSP_ARCH=$DspArch FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF $matmulCmakeDefines"
  # The original experiment fixed the environment script at
  # /root/llama-npu-env.sh. Keep that as the parameter default while allowing
  # clean-clone users to select their own WSL setup.
  $cmd = "set -eo pipefail; source '$WslEnvScript' >/dev/null 2>&1; cd '$htpWsl'; $matmulEnvExports; build_cmake android; build_cmake hexagon DSP_ARCH=$DspArch FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF"
  Invoke-WslLogged $cmd
}

if (-not $SkipLlama) {
  $llamaWsl = ConvertTo-WslPath $LlamaRoot
  $cachePath = Join-Path $LlamaBuildDir "CMakeCache.txt"
  $cmakeFilesPath = Join-Path $LlamaBuildDir "CMakeFiles"
  $staleCache = $false
  if (Test-Path -LiteralPath $cachePath) {
    $homeLine = Select-String -LiteralPath $cachePath -Pattern "^CMAKE_HOME_DIRECTORY:INTERNAL=" -SimpleMatch:$false | Select-Object -First 1
    if ($homeLine) {
      $cachedHome = ($homeLine.Line -split "=", 2)[1]
      if ($cachedHome -ne $llamaWsl) {
        $staleCache = $true
        Write-Host "Detected stale llama CMake cache source: $cachedHome"
        Write-Host "Expected llama source: $llamaWsl"
      }
    }
  }
  if (($ForceConfigure -or $staleCache) -and (Test-Path -LiteralPath $LlamaBuildDir)) {
    $resolvedBuild = (Resolve-Path -LiteralPath $LlamaBuildDir).Path
    $resolvedRoot = (Resolve-Path -LiteralPath $LlamaRoot).Path
    if (-not $resolvedBuild.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
      throw "refusing to clean generated CMake cache outside isolated llama tree: $resolvedBuild"
    }
    # LPBQ deploy-v1 note: this directory is copied from earlier artifacts, so
    # generated CMake metadata can point at the old experiment source. The old
    # metadata-only cleanup is kept as a rollback breadcrumb, but it can leave a
    # stale build.ninja next to a fresh CMakeFiles tree and break compiler tests.
    # Remove-Item -LiteralPath $cachePath -Force -ErrorAction SilentlyContinue
    # Remove-Item -LiteralPath $cmakeFilesPath -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force -ErrorAction SilentlyContinue
  }
  $configure = $ForceConfigure -or -not (Test-Path -LiteralPath (Join-Path $LlamaBuildDir "CMakeCache.txt"))
  if ($configure) {
    # The old SDK 6.3.0.0 path remains the default via AndroidNdkRootWsl.
    $cmd = "set -eo pipefail; source '$WslEnvScript' >/dev/null 2>&1; cd '$llamaWsl'; cmake -S . -B '$BuildDirName' -G Ninja -DCMAKE_TOOLCHAIN_FILE='$AndroidNdkRootWsl/build/cmake/android.toolchain.cmake' -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DGGML_HTP=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release"
    Invoke-WslLogged $cmd
  }
  $cmd = "set -eo pipefail; source '$WslEnvScript' >/dev/null 2>&1; cd '$llamaWsl'; cmake --build '$BuildDirName' --target llama-server llama-cli llama ggml ggml-base ggml-cpu ggml-htp -j $Jobs"
  Invoke-WslLogged $cmd
}

if (-not $SkipPackage) {
  try {
    $modelPath = Ensure-LocalHmxFp16Model -ModelPath $ModelPath
  } catch {
    # LPBQ deploy-v1 packages runtime binaries from the isolated build tree.
    # The legacy pure-FP16 model source may have been moved, so keep packaging
    # usable when only runtime artifacts are needed.
    Write-Warning "Skipping FP16 model manifest refresh: $($_.Exception.Message)"
    $modelPath = $null
  }
  $HtpAndroidShip = Join-Path $HtpRoot "android_ReleaseG_aarch64\ship"
  $HtpDspShip = Join-Path $HtpRoot "hexagon_ReleaseG_toolv88_$DspArch\ship"
  $runtimeFiles = @(
    @{ Source = (Join-Path $LlamaBuildDir "bin\llama-server"); Dest = "llama-server" },
    @{ Source = (Join-Path $LlamaBuildDir "bin\llama-cli"); Dest = "llama-cli" },
    @{ Source = (Join-Path $LlamaBuildDir "ggml\src\libggml.so"); Dest = "libggml.so" },
    @{ Source = (Join-Path $LlamaBuildDir "ggml\src\libggml-base.so"); Dest = "libggml-base.so" },
    @{ Source = (Join-Path $LlamaBuildDir "ggml\src\libggml-cpu.so"); Dest = "libggml-cpu.so" },
    @{ Source = (Join-Path $LlamaBuildDir "ggml\src\ggml-htp\libggml-htp.so"); Dest = "libggml-htp.so" },
    @{ Source = (Join-Path $LlamaBuildDir "src\libllama.so"); Dest = "libllama.so" },
    @{ Source = (Join-Path $HtpAndroidShip "htp_ops_test"); Dest = "htp_ops_test" },
    @{ Source = (Join-Path $HtpAndroidShip "libhtp_ops.so"); Dest = "libhtp_ops.so" },
    @{ Source = (Join-Path $HtpDspShip "libhtp_ops_skel.so"); Dest = "libhtp_ops_skel.so" },
    @{ Source = (Join-Path $DeployDir "run_server.sh"); Dest = "run_server.sh" }
  )
  foreach ($file in $runtimeFiles) {
    if (-not (Test-Path -LiteralPath $file.Source)) {
      throw "missing build output: $($file.Source)"
    }
    $destPath = Join-Path $DeployDir $file.Dest
    if ((Resolve-Path -LiteralPath $file.Source).Path -ne (Resolve-Path -LiteralPath $destPath -ErrorAction SilentlyContinue).Path) {
      Copy-Item -LiteralPath $file.Source -Destination $destPath -Force
    }
  }
  New-Item -ItemType Directory -Force -Path (Join-Path $DeployDir "cdsp"), (Join-Path $DeployDir "dsp") | Out-Null
  Copy-Item -LiteralPath (Join-Path $HtpDspShip "libhtp_ops_skel.so") -Destination (Join-Path $DeployDir "cdsp\libhtp_ops_skel.so") -Force
  Copy-Item -LiteralPath (Join-Path $HtpDspShip "libhtp_ops_skel.so") -Destination (Join-Path $DeployDir "dsp\libhtp_ops_skel.so") -Force

  $manifest = [ordered]@{
    built_at = (Get-Date).ToString("o")
    dsp_arch = $DspArch
    fp16_weight_load_mode = $Fp16WeightLoadMode
    fp16_weight_load_mode_value = $Fp16WeightLoadModeValue
    fp16_dma_dst_bypass = $Fp16DmaDstBypass
    fp16_direct_final_dma = $Fp16DirectFinalDma
    fp16_direct_final_pipeline = $Fp16DirectFinalPipeline
    fp16_direct_final_cache_invalidate = $Fp16DirectFinalCacheInvalidate
    fp16_direct_final_touch = $Fp16DirectFinalTouch
    fp16_decode_direct_pipeline = $Fp16DecodeDirectPipeline
    fp16_decode_pipeline_max_m = $Fp16DecodePipelineMaxM
    fp16_parallel_weight_publish = $Fp16ParallelWeightPublish
    fp16_publish_use_memcpy = $Fp16PublishUseMemcpy
    fp16_parallel_output_store = $Fp16ParallelOutputStore
    fp16_small_m_vtcm = [ordered]@{
      enabled = $Fp16SmallMVtcmEnable
      threshold = $Fp16SmallMThreshold
      weight = $Fp16SmallMWeightKb
      activation = $Fp16SmallMActivationKb
      output = $Fp16SmallMOutputKb
      scratch = $Fp16SmallMScratchKb
      pipeline_output = $Fp16SmallMPipelineOutputKb
    }
    matmul_vtcm_kb = [ordered]@{
      weight = $MatmulWeightKb
      activation = $MatmulActivationKb
      output = $MatmulOutputKb
      scratch = $MatmulScratchKb
      pipeline_output = $MatmulPipelineOutputKb
    }
    fp16_os_block = [ordered]@{
      m = $Fp16OsMBlock
      n = $Fp16OsNBlock
      k = $Fp16OsKBlock
    }
    fp16_os_weight_gather = [ordered]@{
      direct_dma = $Fp16OsWeightGatherDma
      dma_scratch = $Fp16OsWeightGatherDmaScratch
      prefill_only = $Fp16OsWeightGatherDmaPrefillOnly
    }
    matmul_pipeline_mode = $MatmulPipelineMode
    lpbq_full_u8_v6_copy_1024_loop8 = $LpbqFullU8V6Copy1024Loop8
    lpbq_full_u8_v6_copy_bulk_loop8 = $LpbqFullU8V6CopyBulkLoop8
    lpbq_hvx_weight_copy_prefetch = $LpbqHvxWeightCopyPrefetch
    lpbq_hvx_weight_copy_prefetch_bytes = $LpbqHvxWeightCopyPrefetchBytes
    lpbq_kmajor_weight_group_copy = $LpbqKmajorWeightGroupCopy
    lpbq_kmajor_weight_dma_load = $LpbqKmajorWeightDmaLoad
    lpbq_kmajor_weight_dma_dst_bypass = $LpbqKmajorWeightDmaDstBypass
    lpbq_kmajor_weight_dma_src_bypass = $LpbqKmajorWeightDmaSrcBypass
    lpbq_kmajor_weight_dma_preissue_wait = $LpbqKmajorWeightDmaPreissueWait
    lpbq_kmajor_weight_dma_visibility_sync = $LpbqKmajorWeightDmaVisibilitySync
    lpbq_kmajor_weight_dma_overlap_act = $LpbqKmajorWeightDmaOverlapAct
    lpbq_kmajor_weight_dma_double_buffer = $LpbqKmajorWeightDmaDoubleBuffer
    lpbq_kmajor_weight_dma_n_block_tiles = $LpbqKmajorWeightDmaNBlockTiles
    lpbq_kmajor_weight_dma_overlap_rowblock = $LpbqKmajorWeightDmaOverlapRowblock
    lpbq_ffndown_decode_g32_staging_ab = $LpbqFfnDownDecodeG32StagingAb
    lpbq_ffngateup_decode_g32_staging_ab = $LpbqFfnGateUpDecodeG32StagingAb
    lpbq_weight_publish_trace_split = $LpbqWeightPublishTraceSplit
    lpbq_weight_path_trace_split = $LpbqWeightPathTraceSplit
    lpbq_trace_w0_expert_buckets = $LpbqTraceW0ExpertBuckets
    lpbq_r4_fwht_v2 = $LpbqR4FwhtV2
    lpbq_r4_fwht_direct_v6_store = $LpbqR4FwhtDirectV6Store
    lpbq_r4_fwht_block = $LpbqR4FwhtBlock
    lpbq_r4_fwht_scale_granularity = $LpbqR4FwhtScaleGranularity
    lpbq_act_rows1_direct_v6 = $LpbqActRows1DirectV6
    lpbq_act_packet_reuse = $LpbqActPacketReuse
    lpbq_act_static_scale_nonr4 = $LpbqActStaticScaleNonR4
    lpbq_hmx_convert_affine_bench = $LpbqHmxConvertAffineBench
    lpbq_hmx_reuse_scale_payload_per_ntile = $LpbqHmxReuseScalePayloadPerNTile
    lpbq_hmx_hold_acquire_per_rowblock = $LpbqHmxHoldAcquirePerRowblock
    lpbq_hmx_batch_issue_kgroups = $LpbqHmxBatchIssueKgroups
    lpbq_hmx_double_buffer_drain = $LpbqHmxDoubleBufferDrain
    lpbq_hmx_epilogue_lag_one_tile = $LpbqHmxEpilogueLagOneTile
    lpbq_hmx_opcode_corpus = $LpbqHmxOpcodeCorpus
    lpbq_hmx_after_issue_hvx_probe = $LpbqHmxAfterIssueHvxProbe
    lpbq_hmx_after_issue_hvx_probe_iters = $LpbqHmxAfterIssueHvxProbeIters
    lpbq_full_v6_weight_dma_load = $LpbqFullV6WeightDmaLoad
    lpbq_full_v6_weight_dma_scratch = $LpbqFullV6WeightDmaScratch
    lpbq_full_v6_weight_dma_dst_bypass = $LpbqFullV6WeightDmaDstBypass
    lpbq_full_v6_weight_dma_overlap_act = $LpbqFullV6WeightDmaOverlapAct
    lpbq_full_v6_weight_direct_mxmem_load = $LpbqFullV6WeightDirectMxmemLoad
    lpbq_full_v6_weight_publish_pingpong = $LpbqFullV6WeightPublishPingpong
    lpbq_full_v6_weight_publish_pingpong_small_m_max = $LpbqFullV6WeightPublishPingpongSmallMMax
    lpbq_full_v6_weight_parallel_publish = $LpbqFullV6WeightParallelPublish
    lpbq_full_v6_weight_async_prepublish_act = $LpbqFullV6WeightAsyncPrepublishAct
    lpbq_full_v6_macro_kgroup_pipeline = $LpbqFullV6MacroKgroupPipeline
    lpbq_activation_hmx_cache = $LpbqActivationHmxCache
    lpbq_activation_hmx_cache_max_kb = $LpbqActivationHmxCacheMaxKb
    lpbq_activation_hvx_quant = $LpbqActivationHvxQuant
    lpbq_activation_fp16_staged_quant = $LpbqActivationFp16StagedQuant
    lpbq_disable_online_input_scale = $LpbqDisableOnlineInputScale
    lpbq_activation_hvx_vector_round_store = $LpbqActivationHvxVectorRoundStore
    lpbq_activation_hvx_floor_round_store = $LpbqActivationHvxFloorRoundStore
    lpbq_r4_full_u8_k32_scale_bits_override = $LpbqR4FullU8K32ScaleBitsOverride
    lpbq_r4_full_u8_k32_recover_shift_override = $LpbqR4FullU8K32RecoverShiftOverride
    lpbq_full_u8_precompensate_hmx_scale = $LpbqFullU8PrecompensateHmxScale
    lpbq_full_u8_stream_v6_group_tiles = $LpbqFullU8StreamV6GroupTiles
    lpbq_r4_full_u8_stream_v6_group_tiles = $LpbqR4FullU8StreamV6GroupTiles
    lpbq_full_u8_stream_v6_group_compact_weight = $LpbqFullU8StreamV6GroupCompactWeight
    lpbq_full_u8_stream_v6_group_n_tile_overlap = $LpbqFullU8StreamV6GroupNTileOverlap
    lpbq_full_u8_stream_v6_group_n_tile_overlap_small_m_only = $LpbqFullU8StreamV6GroupNTileOverlapSmallMOnly
    lpbq_full_u8_stream_v6_group_n_tile_overlap_prefill_only = $LpbqFullU8StreamV6GroupNTileOverlapPrefillOnly
    lpbq_full_u8_stream_v6_group_n_tile_overlap_acc_final_issue = $LpbqFullU8StreamV6GroupNTileOverlapAccFinalIssue
    lpbq_skip_unused_full_u8_hi_act_clear = $LpbqSkipUnusedFullU8HiActClear
    lpbq_r4_full_u8_stream_v6_small_m_max = $LpbqR4FullU8StreamV6SmallMMax
    lpbq_r4_full_u8_stream_v6_min_m = $LpbqR4FullU8StreamV6MinM
    lpbq_r4_parallel_prefill = $LpbqR4ParallelPrefill
    lpbq_r4_cached_x4_kk_outer = $LpbqR4CachedX4KkOuter
    lpbq_r4_hvx_dot = $LpbqR4HvxDot
    lpbq_r4_hvx_dot16 = $LpbqR4HvxDot16
    lpbq_r4_dot4_rowpair = $LpbqR4Dot4Rowpair
    lpbq_r4_fp16_hmx_rotate = $LpbqR4Fp16HmxRotate
    lpbq_r4_prefill_hmx_dense = $LpbqR4PrefillHmxDense
    lpbq_r4_prefill_hmx_min_m = $LpbqR4PrefillHmxMinM
    lpbq_r4_hmx_dense_block = $LpbqR4HmxDenseBlock
    lpbq_r4_hmx_highbyte_direct_v6_probe = $LpbqR4HmxHighbyteDirectV6Probe
    lpbq_r4_hmx_hfscale_direct_v6_probe = $LpbqR4HmxHfscaleDirectV6Probe
    lpbq_r4_hmx_hfscale_vtcm_weight_cache_probe = $LpbqR4HmxHfscaleVtcmWeightCacheProbe
    lpbq_r4_hmx_hfscale_vtcm_weight_cache_groups = $LpbqR4HmxHfscaleVtcmWeightCacheGroups
    lpbq_r4_v6_k128_bulk = $LpbqR4V6K128Bulk
    lpbq_r4_v6_k128_parallel_rows = $LpbqR4V6K128ParallelRows
    lpbq_r4_v6_k128_skip_inner_clear = $LpbqR4V6K128SkipInnerClear
    lpbq_r4_v6_k128_bulk_small_m = $LpbqR4V6K128BulkSmallM
    lpbq_r4_v6_k128_bulk_rowpair = $LpbqR4V6K128BulkRowpair
    lpbq_r4_sign_fht_fast = $LpbqR4SignFhtFast
    lpbq_r4_standard_fht_force = $LpbqR4StandardFhtForce
    lpbq_r4_standard_fht_hmx_probe = $LpbqR4StandardFhtHmxProbe
    lpbq_r4_standard_fht_hmx_integrated = $LpbqR4StandardFhtHmxIntegrated
    lpbq_r4_standard_fht_hvx_probe = $LpbqR4StandardFhtHvxProbe
    lpbq_r4_standard_fht_hvx_integrated = $LpbqR4StandardFhtHvxIntegrated
    lpbq_r4_standard_fht_hvx_nibble_probe = $LpbqR4StandardFhtHvxNibbleProbe
    lpbq_r4_standard_fht_hvx_nibble_integrated = $LpbqR4StandardFhtHvxNibbleIntegrated
    lpbq_r4_standard_fht_core_probe = $LpbqR4StandardFhtCoreProbe
    lpbq_r4_standard_fht_rowpair_direct = $LpbqR4StandardFhtRowpairDirect
    lpbq_r4_standard_fht_i16_direct = $LpbqR4StandardFhtI16Direct
    lpbq_r4_standard_fht_hvx_full_probe = $LpbqR4StandardFhtHvxFullProbe
    lpbq_r4_standard_fht_hvx_full_integrated = $LpbqR4StandardFhtHvxFullIntegrated
    lpbq_r4_standard_fht_fp16_direct = $LpbqR4StandardFhtFp16Direct
    lpbq_r4_standard_fht_hvx_full_nibble_probe = $LpbqR4StandardFhtHvxFullNibbleProbe
    lpbq_r4_standard_fht_hvx_full_nibble_integrated = $LpbqR4StandardFhtHvxFullNibbleIntegrated
    lpbq_r4_sign_fht_approx = $LpbqR4SignFhtApprox
    lpbq_r4_sign_fht_colscale_approx = $LpbqR4SignFhtColScaleApprox
    lpbq_r4_prefetch_cols8 = $LpbqR4PrefetchCols8
    lpbq_r4_prefetch_k128_rowpar_cols8 = $LpbqR4PrefetchK128RowparCols8
    # LPBQ deploy-v1 2026-06-17: old nibble-store8 macro stays disabled while
    # lpbq_r4_u8_v6_quant_store8_fast records the narrow U8 V6 candidate.
    lpbq_r4_quant_store8_fast = 0
    lpbq_r4_u8_v6_quant_store8_fast = $LpbqR4QuantStore8Fast
    lpbq_r4_rows1_reduce8_fast = $LpbqR4Rows1Reduce8Fast
    lpbq_r4_rows1_dot4_split = $LpbqR4Rows1Dot4Split
    lpbq_r4_dot8_r128_unroll = $LpbqR4Dot8R128Unroll
    lpbq_r4_dot_pack_precompute_row_store = $LpbqR4DotPackPrecomputeRowStore
    lpbq_r4_k128_dot_pack_precompute_row_store = $LpbqR4K128DotPackPrecomputeRowStore
    lpbq_r4_v6_k128_rowgroup4_dot2 = $LpbqR4V6K128Rowgroup4Dot2
    lpbq_r4_v6_k128_colgroup_parallel = $LpbqR4V6K128ColgroupParallel
    lpbq_r4_v6_k128_workers_max = $LpbqR4V6K128WorkersMax
    lpbq_r4_v6_k128_skip_acc_zero = $LpbqR4V6K128SkipAccZero
    lpbq_r4_v6_k128_rowpair_r128_unroll = $LpbqR4V6K128RowpairR128Unroll
    lpbq_r4_v6_k128_rowpair_dot8 = $LpbqR4V6K128RowpairDot8
    lpbq_r4_v6_k128_rowpair_reduce8 = $LpbqR4V6K128RowpairReduce8
    lpbq_r4_v6_k128_rowpair_store2_fast = $LpbqR4V6K128RowpairStore2Fast
    lpbq_r4_v6_k128_rowpair_fused_store4 = $LpbqR4V6K128RowpairFusedStore4
    lpbq_r4_v6_k128_rowpair_fused_store4_r128_unroll = $LpbqR4V6K128RowpairFusedStore4R128Unroll
    lpbq_r4_v6_k128_rowpair_aligned_loads = $LpbqR4V6K128RowpairAlignedLoads
    lpbq_r4_v6_k128_rowpair_prebase_store8 = $LpbqR4V6K128RowpairPrebaseStore8
    lpbq_r4_v6_k128_rowpair_const_lanes = $LpbqR4V6K128RowpairConstLanes
    lpbq_r4_scale_cache_r128_unroll = $LpbqR4ScaleCacheR128Unroll
    lpbq_r4_trace_dot_pack_split = $LpbqR4TraceDotPackSplit
    lpbq_r4_nibble_issue_acc_overlap = $LpbqR4NibbleIssueAccOverlap
    lpbq_r4_nibble_active_row_accum = $LpbqR4NibbleActiveRowAccum
    lpbq_r4_nibble_raw_accum = $LpbqR4NibbleRawAccum
    lpbq_r4_nibble_raw_accum_overlap = $LpbqR4NibbleRawAccumOverlap
    # LPBQ deploy-v1 2026-07-03: expose the two R4 full-U8 route gates in the
    # manifest so full-V6 real-layer ret=-16012 can be diagnosed without
    # re-opening the build log.
    lpbq_r4_full_u8_singlepass = $LpbqR4FullU8Singlepass
    lpbq_r4_full_u8_stream_v6 = $LpbqR4FullU8StreamV6
    lpbq_r4_full_u8_singlepass_recover_round_value = $LpbqR4FullU8SinglepassRecoverRoundValue
    lpbq_r4_full_u8_singlepass_recover_round_value_rows_gt1 = $LpbqR4FullU8SinglepassRecoverRoundValueRowsGt1
    lpbq_r4_full_u8_singlepass_recover_apply_257 = $LpbqR4FullU8SinglepassRecoverApply257
    lpbq_r4_full_u8_singlepass_recover_neg_extra = $LpbqR4FullU8SinglepassRecoverNegExtra
    lpbq_r4_full_u8_singlepass_recover_corr_rshift = $LpbqR4FullU8SinglepassRecoverCorrRshift
    lpbq_r4_full_u8_singlepass_recover_corr_rshift_rows_gt1 = $LpbqR4FullU8SinglepassRecoverCorrRshiftRowsGt1
    lpbq_r4_full_u8_singlepass_recover_corr_extra_rshift = $LpbqR4FullU8SinglepassRecoverCorrExtraRshift
    lpbq_r4_full_u8_singlepass_recover_corr_extra_rshift_rows_gt1 = $LpbqR4FullU8SinglepassRecoverCorrExtraRshiftRowsGt1
    lpbq_r4_full_u8_stream_v6_group_sign_compensate = $LpbqR4FullU8StreamV6GroupSignCompensate
    lpbq_r4_full_u8_stream_v6_group_exact_recover = $LpbqR4FullU8StreamV6GroupExactRecover
    lpbq_full_u8_stream_v6_group_exact_recover = $LpbqFullU8StreamV6GroupExactRecover
    lpbq_r4_full_u8_stream_v6_group_adaptive_scale = $LpbqR4FullU8StreamV6GroupAdaptiveScale
    lpbq_r4_full_u8_stream_v6_group_adaptive_raw_adjust = $LpbqR4FullU8StreamV6GroupAdaptiveRawAdjust
    lpbq_r4_full_u8_stream_v6_group_adaptive_round_bias_bits = "$LpbqR4FullU8StreamV6GroupAdaptiveRoundBiasBits"
    lpbq_r4_accum_decode_unroll = $LpbqR4AccumDecodeUnroll
    lpbq_r4_structured_fwht = $LpbqR4StructuredFwht
    lpbq_r4_structured_fwht_stage_unroll = $LpbqR4StructuredFwhtStageUnroll
    lpbq_r4_structured_fwht_direct_k128 = $LpbqR4StructuredFwhtDirectK128
    lpbq_adaptive_bucket_fast_0_3 = $LpbqAdaptiveBucketFast03
    lpbq_adaptive_bucket_fast_0_3_rows1 = $LpbqAdaptiveBucketFast03Rows1
    lpbq_adaptive_accum_varshift_rows1 = $LpbqAdaptiveAccumVarshiftRows1
    lpbq_adaptive_shift0_fast_path = $LpbqAdaptiveShift0FastPath
    lpbq_adaptive_uniform_shift0_noshift = $LpbqAdaptiveUniformShift0NoShift
    lpbq_adaptive_shift0_singlepass = $LpbqAdaptiveShift0Singlepass
    lpbq_adaptive_probe_scale_cache = $LpbqAdaptiveProbeScaleCache
    lpbq_adaptive_probe_only_shift5_fast = $LpbqAdaptiveProbeOnlyShift5Fast
    lpbq_adaptive_probe_scalar_small_rows = $LpbqAdaptiveProbeScalarSmallRows
    lpbq_adaptive_retry_skip_init = $LpbqAdaptiveRetrySkipInit
    lpbq_adaptive_scale_low32_only = $LpbqAdaptiveScaleLow32Only
    lpbq_adaptive_scale_skip_unchanged = $LpbqAdaptiveScaleSkipUnchanged
    lpbq_adaptive_derive_i32_max = $LpbqAdaptiveDeriveI32Max
    lpbq_adaptive_scale_word_lut = $LpbqAdaptiveScaleWordLut
    lpbq_adaptive_derive_row0_exact_rows1 = $LpbqAdaptiveDeriveRow0ExactRows1
    lpbq_r4_grouped_v6_rowblock_weight_reuse = $LpbqR4GroupedV6RowblockWeightReuse
    lpbq_r4_grouped_v6_rowblock_weight_reuse_blocks = $LpbqR4GroupedV6RowblockWeightReuseBlocks
    lpbq_nonr4_grouped_v6_rowblock_weight_reuse = $LpbqNonR4GroupedV6RowblockWeightReuse
    lpbq_nonr4_grouped_v6_rowblock_weight_reuse_blocks = $LpbqNonR4GroupedV6RowblockWeightReuseBlocks
    lpbq_nonr4_grouped_v6_nblock_rowblock_reuse = $LpbqNonR4GroupedV6NBlockRowblockReuse
    lpbq_nonr4_grouped_v6_nblock_rowblock_reuse_tiles = $LpbqNonR4GroupedV6NBlockRowblockReuseTiles
    lpbq_r4_grouped_v6_rowblock_hmx_acquire_per_kgroup = $LpbqR4GroupedV6RowblockHmxAcquirePerKgroup
    lpbq_r4_rowblock_probe_only_shift5_fast = $LpbqR4RowblockProbeOnlyShift5Fast
    lpbq_r4_rowblock_skip_initial_scale_load = $LpbqR4RowblockSkipInitialScaleLoad
    lpbq_hmx_fine_subtrace = $LpbqHmxFineSubtrace
    lpbq_full_u8_rows1_accum_pair_unroll = $LpbqFullU8Rows1AccumPairUnroll
    lpbq_scaled_k128_groups = $LpbqScaledK128Groups
    lpbq_scaled_k_group_tiles = $LpbqScaledKGroupTiles
    lpbq_r4_scaled_k_groups = $LpbqR4ScaledKGroups
    lpbq_exact_k64_safe_groups = $LpbqExactK64SafeGroups
    lpbq_exact_k_safe_group_tiles = $LpbqExactKSafeGroupTiles
    lpbq_r4_exact_k64_safe_groups = $LpbqR4ExactK64SafeGroups
    lpbq_exact_k64_non_r4_groups = $LpbqExactK64NonR4Groups
    lpbq_exact_k64_safe_abs_limit = $LpbqExactK64SafeAbsLimit
    lpbq_exact_k64_debug_stop_stage = $LpbqExactK64DebugStopStage
    lpbq_mode166_stop_after = $LpbqMode166StopAfter
    lpbq_mode166_v6_one_stop_stage = $LpbqMode166V6OneStopStage
    lpbq_exact_k64_min_safe_percent = $LpbqExactK64MinSafePercent
    lpbq_exact_k64_acquire_per_group = $LpbqExactK64AcquirePerGroup
    lpbq_exact_k64_issue_acc_overlap = $LpbqExactK64IssueAccOverlap
    lpbq_exact_k64_full_weight_stream = $LpbqExactK64FullWeightStream
    lpbq_exact_k64_pair_preaccum = $LpbqExactK64PairPreaccum
    lpbq_exact_k64_require_all_safe = $LpbqExactK64RequireAllSafe
    lpbq_exact_k64_force_unsafe = $LpbqExactK64ForceUnsafe
    lpbq_hmx_clear_out_before_store = $LpbqHmxClearOutBeforeStore
    lpbq_hmx_clear_active_rows_only = $LpbqHmxClearActiveRowsOnly
    lpbq_hmx_cvt_uh_drain = $LpbqHmxCvtUhDrain
    lpbq_hmx_cvt_uh_selector2_drain = $LpbqHmxCvtUhSelector2Drain
    lpbq_full_u8_acquire_per_m = $LpbqFullU8AcquirePerM
    lpbq_hmx_acquire_per_m = $LpbqHmxAcquirePerM
    lpbq_immutable_sidecar_cache = $LpbqImmutableSidecarCache
    lpbq_roundhalf_folded_store_fast = $LpbqRoundhalfFoldedStoreFast
    lpbq_roundhalf_folded_raw_adjust_bias_fast = $LpbqRoundhalfFoldedRawAdjustBiasFast
    build_log = $LogPath
    model = $modelPath
    llama_build_dir = $LlamaBuildDir
    htp_android_ship = $HtpAndroidShip
    htp_dsp_ship = $HtpDspShip
    files = $runtimeFiles | ForEach-Object {
      [ordered]@{ name = $_.Dest; source = $_.Source; size = (Get-Item -LiteralPath $_.Source).Length }
    }
  }
  Write-JsonFile -Object $manifest -Path (Join-Path $DeployDir "deploy_manifest.json") -Depth 16
}

Write-Host "Build/package complete. Log: $LogPath"
