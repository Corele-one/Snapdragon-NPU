#include "dsp/hmx_mgr.h"
#include "dsp/worker_pool.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

static int hmx_mgr_ctx_id;
static int hmx_mgr_spin_lock;
static int hmx_mgr_lock_mode = HAP_COMPUTE_RES_HMX_SHARED;

// HMX enable/disable calls are nested on the message receiver thread.  V81
// devices that only expose the exclusive lock must not issue a second lock for
// the inner benchmark scope.
static __thread int hmx_mgr_thread_lock_depth;

worker_pool_context_t hmx_worker_pool_ctx; 

void hmx_manager_setup() {
  // NOTE(hzx): HMX should be already powered up in power_setup()

  // A prior host process can time out while a diagnostic kernel owns this
  // process-local lock.  Reset it before accepting work in a newly opened
  // FastRPC session; the hardware ownership is still governed by HAP below.
  hmx_mgr_spin_lock = 0;

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);
  HAP_compute_res_attr_set_hmx_param(&req, 1);

  hmx_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 10ms timeout
  if (hmx_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    return;
  }

  // The historical implementation always used SHARED.  Snapdragon 8 Elite
  // Gen 5 reports that SHARED is unsupported, so probe once and select the
  // lock mode before any worker thread can retain it for its lifetime.
  int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
  if (err == 0) {
    HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
    hmx_mgr_lock_mode = HAP_COMPUTE_RES_HMX_SHARED;
  } else {
    // SDK 6.6 names the exclusive ownership mode NON_SHARED.  Keep the
    // historical SHARED branch above for older runtimes, but use the declared
    // NON_SHARED enum when a V81 runtime rejects SHARED.
    hmx_mgr_lock_mode = HAP_COMPUTE_RES_HMX_NON_SHARED;
    FARF(ALWAYS, "%s: SHARED HMX lock unavailable; using NON_SHARED", __func__);
  }

  if (hmx_mgr_lock_mode == HAP_COMPUTE_RES_HMX_SHARED) {
    err = worker_pool_init_ex(&hmx_worker_pool_ctx, 8192, 1, 1);
    if (err) {
      FARF(ALWAYS, "%s: HMX worker pool init failed", __func__);
    }
  } else {
    // The current worker keeps its HMX lock for the lifetime of the thread.
    // That is valid for SHARED mode but would starve the message receiver in
    // NON_SHARED mode.  Roofline kernels run on the receiver thread, so leave
    // the HMX worker pool disabled for this runtime capability.
    hmx_worker_pool_ctx = NULL;
  }
}

void hmx_manager_reset() {
  if (hmx_worker_pool_ctx) {
    worker_pool_deinit(&hmx_worker_pool_ctx);
  }

  if (hmx_mgr_ctx_id) {
    HAP_compute_res_release(hmx_mgr_ctx_id);
  }

  hmx_mgr_ctx_id = 0;
  hmx_mgr_spin_lock = 0;
  hmx_mgr_lock_mode = HAP_COMPUTE_RES_HMX_SHARED;
  hmx_mgr_thread_lock_depth = 0;
}

void hmx_manager_enable_execution() {
  if (!hmx_mgr_ctx_id) {
    return;
  }

  if (hmx_mgr_thread_lock_depth > 0) {
    ++hmx_mgr_thread_lock_depth;
    return;
  }

  // Historical code requested HAP_COMPUTE_RES_HMX_SHARED unconditionally.
  // Use the capability selected during setup so V81-only exclusive runtimes
  // hold a valid lock before the first matrix instruction is issued.
  int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, hmx_mgr_lock_mode);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_hmx_lock2(mode=%d) failed with return code 0x%x", hmx_mgr_lock_mode, err);
    return;
  }
  hmx_mgr_thread_lock_depth = 1;
}

void hmx_manager_disable_execution() {
  if (!hmx_mgr_ctx_id || hmx_mgr_thread_lock_depth <= 0) {
    return;
  }

  if (--hmx_mgr_thread_lock_depth > 0) {
    return;
  }

  HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, hmx_mgr_lock_mode);
}

void hmx_unit_acquire() {
  int *lock_ptr = &hmx_mgr_spin_lock;
  asm volatile(
    "1:  r0 = memw_locked(%0)     \n"
    "    p0 = cmp.eq(r0, #0)      \n"
    "    if (!p0) jump 2f         \n"
    "    memw_locked(%0, p0) = %0 \n"
    "    if (p0) jump 3f          \n"
    "2:  pause(#8)                \n"
    "    jump 1b                  \n"
    "3:"
    : "+r"(lock_ptr)::"p0", "r0");
}

void hmx_unit_release() {
  *(volatile int *) &hmx_mgr_spin_lock = 0;
}
