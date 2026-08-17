#include <HAP_compute_res.h>

/*
 * The SDK 6.6 headers expose compute_resource_hmx_lock2(), while the v79
 * run_main_on_hexagon_sim image exports only the original lock/unlock ABI.
 * Returning NOT_SUPPORTED for SHARED makes hmx_mgr.c select NON_SHARED; that
 * mode is then adapted to the legacy ABI without touching Attention math.
 */
int compute_resource_hmx_lock2(unsigned int context_id, compute_res_hmx_type_t type) {
  if (type == HAP_COMPUTE_RES_HMX_SHARED || !compute_resource_hmx_lock) {
    return HAP_COMPUTE_RES_NOT_SUPPORTED;
  }
  return compute_resource_hmx_lock(context_id);
}

int compute_resource_hmx_unlock2(unsigned int context_id, compute_res_hmx_type_t type) {
  if (type == HAP_COMPUTE_RES_HMX_SHARED || !compute_resource_hmx_unlock) {
    return HAP_COMPUTE_RES_NOT_SUPPORTED;
  }
  return compute_resource_hmx_unlock(context_id);
}

int scna_sim_hmx_uses_legacy_adapter(void) { return 1; }
