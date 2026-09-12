#include <AEEStdErr.h>
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp_capabilities_utils.h"
#include "scna_env.h"

enum { kPingToken = 0x53434e41 }; /* ASCII: SCNA */

int main(void) {
  const int domain_id = CDSP_DOMAIN_ID;
  domain *target_domain = get_domain(domain_id);
  if (!target_domain) {
    fprintf(stderr, "Cannot resolve cDSP domain %d.\n", domain_id);
    return 1;
  }

  struct remote_rpc_control_unsigned_module unsigned_pd = {
      .domain = domain_id,
      .enable = 1,
  };
  int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd, sizeof(unsigned_pd));
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "Unable to enable unsigned cDSP PD: 0x%08x\n", (unsigned)err);
    return 1;
  }

  const size_t uri_size = strlen(scna_env_URI) + MAX_DOMAIN_URI_SIZE;
  char *uri = malloc(uri_size);
  if (!uri) {
    fprintf(stderr, "Unable to allocate FastRPC URI.\n");
    return 1;
  }
  const int written = snprintf(uri, uri_size, "%s%s", scna_env_URI, target_domain->uri);
  if (written < 0 || (size_t)written >= uri_size) {
    fprintf(stderr, "Unable to construct FastRPC URI.\n");
    free(uri);
    return 1;
  }

  remote_handle64 handle = -1;
  err = scna_env_open(uri, &handle);
  free(uri);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "FastRPC session open failed: 0x%08x\n", (unsigned)err);
    return 1;
  }

  err = scna_env_ping(handle, kPingToken);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "cDSP ping failed: 0x%08x\n", (unsigned)err);
    scna_env_close(handle);
    return 1;
  }

  err = scna_env_close(handle);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "FastRPC session close failed: 0x%08x\n", (unsigned)err);
    return 1;
  }

  puts("SCNA environment smoke test passed.");
  return 0;
}
