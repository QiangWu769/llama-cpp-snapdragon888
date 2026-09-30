#include "dsp/hmx_mgr.h"
#include "dsp/worker_pool.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

static int hmx_mgr_ctx_id;
static int hmx_mgr_spin_lock;

worker_pool_context_t hmx_worker_pool_ctx; 

int hmx_manager_setup() {
  // NOTE(hzx): HMX should be already powered up in power_setup()

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);
  HAP_compute_res_attr_set_hmx_param(&req, 1);

  hmx_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 10ms timeout
  if (hmx_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    return -1;
  }

  int err = worker_pool_init_ex(&hmx_worker_pool_ctx, 8192, 1, 1);
  if (err) {
    FARF(ALWAYS, "%s: HMX worker pool init failed", __func__);
    HAP_compute_res_release(hmx_mgr_ctx_id);
    hmx_mgr_ctx_id = 0;
    return err;
  }
  return 0;
}

void hmx_manager_reset() {
  if (hmx_worker_pool_ctx) {
    worker_pool_deinit(&hmx_worker_pool_ctx);
  }

  if (hmx_mgr_ctx_id) {
    HAP_compute_res_release(hmx_mgr_ctx_id);
    hmx_mgr_ctx_id = 0;
  }
  hmx_mgr_spin_lock = 0;
}

void hmx_manager_enable_execution() {
#if !HTP_HMX_V68
  if (!hmx_mgr_ctx_id) {
    return;
  }

  // enable current thread to timeshare HMX unit
  int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_hmx_lock2 failed with return code 0x%x", err);
  }
#endif
}

void hmx_manager_disable_execution() {
#if !HTP_HMX_V68
  if (!hmx_mgr_ctx_id) {
    return;
  }

  HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
#endif
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
    : "+r"(lock_ptr)::"p0", "r0", "memory");
#if HTP_HMX_V68
  // v68 firmware exports the exclusive API. Lock on the same thread that
  // issues matrix instructions, not once on the thread creating the pool.
  int err = HAP_compute_res_hmx_lock(hmx_mgr_ctx_id);
  if (err) {
    FARF(ALWAYS, "HMX exclusive lock failed: %d", err);
    __builtin_trap();
  }
  asm volatile("barrier" ::: "memory");
#endif
}

void hmx_unit_release() {
  asm volatile("barrier" ::: "memory");
#if HTP_HMX_V68
  int err = HAP_compute_res_hmx_unlock(hmx_mgr_ctx_id);
  if (err) {
    FARF(ALWAYS, "HMX exclusive unlock failed: %d", err);
    __builtin_trap();
  }
#endif
  asm volatile("barrier" ::: "memory");
  *(volatile int *) &hmx_mgr_spin_lock = 0;
}
