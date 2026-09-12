#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <stdint.h>

#include "scna_env.h"

static int scna_env_dummy_handle;

AEEResult scna_env_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  if (handle == NULL) return AEE_EBADPARM;
  *handle = (remote_handle64)&scna_env_dummy_handle;
  return AEE_SUCCESS;
}

/*
 * Keep this skeleton side-effect free.  In particular, do not acquire VTCM or
 * HMX here: the purpose of this project phase is to isolate FastRPC setup from
 * later SCNA kernel/resource work.
 */
AEEResult scna_env_ping(remote_handle64 handle, int32_t token) {
  (void)handle;
  FARF(ALWAYS, "SCNA environment ping: token=0x%x", (unsigned)token);
  return AEE_SUCCESS;
}

AEEResult scna_env_close(remote_handle64 handle) {
  (void)handle;
  return AEE_SUCCESS;
}
