#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

float __mram_noinit w[DIM];
float __mram_noinit x[DIM];

__mram_noinit struct {
  float sum;
  int padding;
} data;

BARRIER_INIT(barrier, NR_TASKLETS);
BARRIER_INIT(reduction_barrier, NR_TASKLETS);

static float square(float x) { return x * x; }

static void add(float *a, float *b) { *a += *b; }

int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  // 1 dpu, 16 tasklets
  __dma_aligned float wram_x[DIM / NR_TASKLETS];
  __dma_aligned float wram_w[DIM / NR_TASKLETS];

  mram_read(x + tasklet_id * (DIM / NR_TASKLETS), wram_x,
            (DIM / NR_TASKLETS) * sizeof(float));
  mram_read(w + tasklet_id * (DIM / NR_TASKLETS), wram_w,
            (DIM / NR_TASKLETS) * sizeof(float));

  // partial reduction
  float ss = 0.0f;
  for (size_t i = 0; i < DIM / NR_TASKLETS; i++) {
    ss += square(wram_x[i]);
  }

  // final reduction
  mram_update_int_atomic((int *)&data.sum, (void (*)(void *, void *))add,
                         (void *)&ss);
  barrier_wait(&reduction_barrier);

  ss = isqrtf(data.sum / DIM + 1e-5f);

  for (size_t i = 0; i < DIM / NR_TASKLETS; i++) {
    wram_x[i] *= wram_w[i] * ss;
  }

  mram_write(wram_x, x + tasklet_id * (DIM / NR_TASKLETS),
             (DIM / NR_TASKLETS) * sizeof(float));

  return 0;
}
