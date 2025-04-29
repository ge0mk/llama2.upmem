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

__mram_noinit float w2[NR_TASKLETS * HIDDEN_DIM];
__mram_noinit float hb[HIDDEN_DIM];
__mram_noinit float x[NR_TASKLETS];

static void add(float *a, float *b) { *a += *b; }

BARRIER_INIT(barrier, NR_TASKLETS);
int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

#define CHUNK_SIZE 256
  float *wram_w = mem_alloc(CHUNK_SIZE * sizeof(float));
  float *wram_h = mem_alloc(CHUNK_SIZE * sizeof(float));

  float r = 0;

  for (size_t i = 0; i < HIDDEN_DIM; i += CHUNK_SIZE) {
    const size_t n = i + CHUNK_SIZE >= HIDDEN_DIM ? HIDDEN_DIM - i : CHUNK_SIZE;
    mram_read(w2 + tasklet_id * HIDDEN_DIM + i, wram_w, n * sizeof(float));
    mram_read(hb + i, wram_h, n * sizeof(float));
    r += dot(wram_w, wram_h, n);
  }

  mram_update_int_atomic((int *)&x[tasklet_id], (void (*)(void *, void *))add,
                         (void *)&r);

  return 0;
}
