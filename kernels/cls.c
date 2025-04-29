#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>

#include <stdint.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

__mram_noinit float x[DIM];
__mram_noinit float wcls[16 * 100 * DIM];
__mram_noinit float logits[16 * 100];

BARRIER_INIT(barrier, NR_TASKLETS);
int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  float *wram_w = mem_alloc(DIM * sizeof(float));
  float *wram_x = mem_alloc(DIM * sizeof(float));
  float *wram_r = mem_alloc(CLS_ROWS_PER_THREAD * sizeof(float));
  mram_read(x, wram_x, DIM * sizeof(float));

  for (size_t i = 0; i < CLS_ROWS_PER_THREAD; i++) {
    size_t offset = tasklet_id * CLS_ROWS_PER_THREAD + i;
    mram_read(wcls + offset * DIM, wram_w, DIM * sizeof(float));
    wram_r[i] = dot(wram_x, wram_w, DIM);
  }

  mram_write(wram_r, logits + tasklet_id * CLS_ROWS_PER_THREAD,
             CLS_ROWS_PER_THREAD * sizeof(float));

  return 0;
}
