#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mram_unaligned.h>

#include <stdint.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

float __mram_noinit x[NR_TASKLETS];
float __mram_noinit xb[DIM];
float __mram_noinit wo[DIM * NR_TASKLETS];

static void add(float *a, float *b) { *a += *b; }

BARRIER_INIT(barrier, NR_TASKLETS);
int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  float *wram_w = mem_alloc(DIM * sizeof(float));
  float *wram_x = mem_alloc(DIM * sizeof(float));

  mram_read(wo + tasklet_id * DIM, wram_w, DIM * sizeof(float));
  mram_read(xb, wram_x, DIM * sizeof(float));
  const float r = dot(wram_w, wram_x, DIM);
  mram_update_int_atomic((int *)&x[tasklet_id], (void (*)(void *, void *))add,
                         (void *)&r);

  return 0;
}
