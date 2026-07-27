#include "dsp/vtcm_mgr.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace vtcm_manager {

uint8_t *vtcm_base           = nullptr;
uint8_t *vtcm_reserved_start = nullptr;
size_t   vtcm_total_size     = 0;

int vtcm_mgr_ctx_id = 0;
int last_query_error = 0;
unsigned int last_query_available_size = 0;
unsigned int last_query_total_size = 0;
unsigned int last_request_size = 0;
int last_acquire_ctx_id = 0;
int last_acquire_attempts = 0;
int last_failure_code = 0;

std::unordered_map<std::string, uint8_t *> reserved_areas;

}  // namespace vtcm_manager

extern "C" {

void vtcm_manager_setup() {
  using namespace vtcm_manager;

  // LPBQ deploy-v1 robustness note: a previous FastRPC crash or still-tearing
  // down PD can make VTCM acquisition fail.  Always clear stale pointers before
  // trying again so later reserve_area() calls fail cleanly instead of forming
  // addresses from a null base.
  vtcm_base = nullptr;
  vtcm_reserved_start = nullptr;
  vtcm_total_size = 0;
  vtcm_mgr_ctx_id = 0;
  last_query_error = 0;
  last_query_available_size = 0;
  last_query_total_size = 0;
  last_request_size = 0;
  last_acquire_ctx_id = 0;
  last_acquire_attempts = 0;
  last_failure_code = 0;
  reserved_areas.clear();

  int err;

  unsigned int            avail_size, total_size;
  compute_res_vtcm_page_t avail_pages, total_pages;
  err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);
  last_query_error = err;
  last_query_available_size = avail_size;
  last_query_total_size = total_size;
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_query_VTCM failed with return code 0x%x", err);
    last_failure_code = 1;
    return;
  }
  FARF(ALWAYS, "available VTCM size: %d KiB, total VTCM size: %d KiB", avail_size / 1024, total_size / 1024);

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);

  // NOTE(hzx): the original path requested all VTCM in one page.  LPBQ deploy
  // keeps that fast path, but after a crashed or concurrently active PD the
  // available size can be smaller than total_size.  Try progressively smaller
  // requests so standalone probes can still diagnose and run when full VTCM is
  // temporarily unavailable.
  unsigned int candidates[7];
  int n_candidates = 0;
  candidates[n_candidates++] = total_size;
  if (avail_size > 0 && avail_size < total_size) {
    candidates[n_candidates++] = avail_size;
  }
  const unsigned int floors[] = {
    4u * 1024u * 1024u,
    2u * 1024u * 1024u,
    1u * 1024u * 1024u,
    512u * 1024u,
    256u * 1024u,
  };
  for (unsigned i = 0; i < sizeof(floors) / sizeof(floors[0]); ++i) {
    if (floors[i] <= total_size) {
      bool seen = false;
      for (int j = 0; j < n_candidates; ++j) {
        if (candidates[j] == floors[i]) {
          seen = true;
          break;
        }
      }
      if (!seen && n_candidates < (int) (sizeof(candidates) / sizeof(candidates[0]))) {
        candidates[n_candidates++] = floors[i];
      }
    }
  }

  unsigned int request_size = 0;
  for (int attempt = 0; attempt < n_candidates; ++attempt) {
    HAP_compute_res_attr_init(&req);
    request_size = candidates[attempt];
    last_request_size = request_size;
    last_acquire_attempts = attempt + 1;
    HAP_compute_res_attr_set_vtcm_param(&req, request_size, 1);
    vtcm_mgr_ctx_id = HAP_compute_res_acquire(&req, 100000);  // timeout 100ms
    if (vtcm_mgr_ctx_id != 0) {
      break;
    }
  }
  last_acquire_ctx_id = vtcm_mgr_ctx_id;
  if (vtcm_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    last_failure_code = 2;
    return;
  }

  vtcm_base = (uint8_t *) HAP_compute_res_attr_get_vtcm_ptr(&req);
  if (!vtcm_base) {
    FARF(ALWAYS, "%s: HAP_compute_res_attr_get_vtcm_ptr returned null", __func__);
    HAP_compute_res_release(vtcm_mgr_ctx_id);
    vtcm_mgr_ctx_id = 0;
    last_failure_code = 3;
    return;
  }
  vtcm_total_size = request_size;
  memset(vtcm_base, 0, request_size);

  vtcm_reserved_start = vtcm_base + request_size;
  last_failure_code = 0;
}

void vtcm_manager_reset() {
  using namespace vtcm_manager;

  if (vtcm_mgr_ctx_id) {
    HAP_compute_res_release(vtcm_mgr_ctx_id);
  }
  vtcm_mgr_ctx_id = 0;
  vtcm_base = nullptr;
  vtcm_reserved_start = nullptr;
  vtcm_total_size = 0;
  reserved_areas.clear();
}

void *vtcm_manager_get_vtcm_base() {
  return vtcm_manager::vtcm_base;
}

size_t vtcm_manager_get_total_size() {
  return vtcm_manager::vtcm_total_size;
}

size_t vtcm_manager_get_seq_capacity() {
  using namespace vtcm_manager;

  if (!vtcm_base || !vtcm_reserved_start || vtcm_reserved_start < vtcm_base) {
    return 0;
  }
  return static_cast<size_t>(vtcm_reserved_start - vtcm_base);
}

int vtcm_manager_get_last_query_error() {
  return vtcm_manager::last_query_error;
}

unsigned int vtcm_manager_get_last_query_available_size() {
  return vtcm_manager::last_query_available_size;
}

unsigned int vtcm_manager_get_last_query_total_size() {
  return vtcm_manager::last_query_total_size;
}

unsigned int vtcm_manager_get_last_request_size() {
  return vtcm_manager::last_request_size;
}

int vtcm_manager_get_last_acquire_ctx_id() {
  return vtcm_manager::last_acquire_ctx_id;
}

int vtcm_manager_get_last_acquire_attempts() {
  return vtcm_manager::last_acquire_attempts;
}

int vtcm_manager_get_last_failure_code() {
  return vtcm_manager::last_failure_code;
}

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment) {
  using namespace vtcm_manager;

  if (!name || size == 0 || !vtcm_base || !vtcm_reserved_start || vtcm_reserved_start <= vtcm_base ||
      (alignment & (alignment - 1)) != 0) {
    FARF(ALWAYS, "%s: VTCM is not ready for reservation '%s' size=%zu", __func__, name ? name : "", size);
    return nullptr;
  }
  if (size > static_cast<size_t>(vtcm_reserved_start - vtcm_base)) {
    FARF(ALWAYS, "%s: VTCM reservation too large for '%s' size=%zu capacity=%zu", __func__, name, size,
         static_cast<size_t>(vtcm_reserved_start - vtcm_base));
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it != reserved_areas.end()) {
    return it->second;
  }

  uintptr_t start_val = reinterpret_cast<uintptr_t>(vtcm_reserved_start - size) & ~(alignment - 1);
  uint8_t  *new_start = reinterpret_cast<uint8_t *>(start_val);
  if (new_start <= vtcm_base) {
    return nullptr;  // no enough space left
  }

  vtcm_reserved_start   = new_start;
  reserved_areas[ident] = new_start;
  return new_start;
}

void *vtcm_manager_query_area(const char *name) {
  using namespace vtcm_manager;

  if (!name) {
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it == reserved_areas.end()) {
    return nullptr;
  }
  return it->second;
}
}
