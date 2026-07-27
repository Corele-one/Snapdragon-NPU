#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vtcm_manager_setup();
void vtcm_manager_reset();

void *vtcm_manager_get_vtcm_base();
size_t vtcm_manager_get_total_size();
size_t vtcm_manager_get_seq_capacity();
int vtcm_manager_get_last_query_error();
unsigned int vtcm_manager_get_last_query_available_size();
unsigned int vtcm_manager_get_last_query_total_size();
unsigned int vtcm_manager_get_last_request_size();
int vtcm_manager_get_last_acquire_ctx_id();
int vtcm_manager_get_last_acquire_attempts();
int vtcm_manager_get_last_failure_code();

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment);
void *vtcm_manager_query_area(const char *name);

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  uint8_t *p = *vtcm_ptr;
  *vtcm_ptr += size;
  return p;
}

#ifdef __cplusplus
}
#endif
