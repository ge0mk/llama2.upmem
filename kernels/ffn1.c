#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>

#include <stdint.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

__mram_noinit float w1[4 * NR_TASKLETS * DIM];
__mram_noinit float w3[4 * NR_TASKLETS * DIM];

__mram_noinit float xb[DIM];
__mram_noinit float hb[4 * NR_TASKLETS];

BARRIER_INIT(barrier, NR_TASKLETS);
int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  float *wram_w = mem_alloc(DIM * sizeof(float));
  float *wram_xb = mem_alloc(DIM * sizeof(float));
  mram_read(xb, wram_xb, DIM * sizeof(float));

  for (size_t i = 0; i < 4; i++) {
    size_t offset = tasklet_id * 4 + i;

    mram_read(w1 + offset * DIM, wram_w, DIM * sizeof(float));
    float h1 = dot(wram_xb, wram_w, DIM);

    mram_read(w3 + offset * DIM, wram_w, DIM * sizeof(float));
    float h2 = dot(wram_xb, wram_w, DIM);

    hb[offset] = h1 * (1.0f / (1.0f + expf(-h1))) * h2;
  }

  return 0;
}
