#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.h"
#include "model_config.h"

float __mram_noinit att[SEQ_LEN];
float __mram_noinit q[HEAD_SIZE];
float __mram_noinit kc[SEQ_LEN * HEAD_SIZE];
float __mram_noinit vc[HEAD_SIZE * SEQ_LEN];
float __mram_noinit x[HEAD_SIZE];

__mram_noinit struct {
  float scale;
  uint32_t pos;
} data;

BARRIER_INIT(barrier, NR_TASKLETS);
BARRIER_INIT(softmax_barrier, NR_TASKLETS);
BARRIER_INIT(softmax_done_barrier, NR_TASKLETS);

int main(void) {
  const size_t tasklet_id = me();
  if (tasklet_id == 0) { // Initialize once the cycle counter
    mem_reset();         // Reset the heap
  }
  barrier_wait(&barrier);

  float *wram_att = mem_alloc(SEQ_LEN * sizeof(float));

  if (tasklet_id < 16) {
    const size_t chunk_size = (SEQ_LEN / 16);
    const size_t chunk_start = tasklet_id * chunk_size;

    float *wram_q = mem_alloc(HEAD_SIZE * sizeof(float));
    float *wram_kc = mem_alloc(HEAD_SIZE * sizeof(float));
    mram_read(q, wram_q, HEAD_SIZE * sizeof(float));

    for (size_t i = 0; i < chunk_size; i++) {
      const size_t t = chunk_start + i;
      if (t <= data.pos) {
        mram_read(kc + t * HEAD_SIZE, wram_kc, HEAD_SIZE * sizeof(float));
        wram_att[i] = dot(wram_q, wram_kc, HEAD_SIZE) / data.scale;
      } else {
        wram_att[i] = -INFINITY;
      }
    }

    mram_write(wram_att, att + chunk_start, chunk_size * sizeof(float));
  }

  barrier_wait(&softmax_barrier);
  if (tasklet_id == 0) {
    mram_read(att, wram_att, SEQ_LEN * sizeof(float));

    float max_val = wram_att[0];
    for (int i = 1; i < SEQ_LEN; i++) {
      if (wram_att[i] > max_val) {
        max_val = wram_att[i];
      }
    }

    float sum = 0.0f;
    for (int i = 0; i < SEQ_LEN; i++) {
      wram_att[i] = expf(wram_att[i] - max_val);
      sum += wram_att[i];
    }

    for (int i = 0; i < SEQ_LEN; i++) {
      wram_att[i] /= sum;
    }

    mram_write(wram_att, att, SEQ_LEN * sizeof(float));
  }
  barrier_wait(&softmax_done_barrier);

  if (tasklet_id < 12) {
    const size_t chunk_size = (HEAD_SIZE / 12);
    const size_t chunk_start = tasklet_id * chunk_size;

    float *wram_vc = mem_alloc(SEQ_LEN * sizeof(float));
    float *wram_x = mem_alloc(chunk_size * sizeof(float));

    mram_read(att, wram_att, SEQ_LEN * sizeof(float));

    for (size_t i = 0; i < chunk_size; i++) {
      mram_read(vc + (chunk_start + i) * SEQ_LEN, wram_vc,
                SEQ_LEN * sizeof(float));
      wram_x[i] = dot(wram_att, wram_vc, SEQ_LEN);
    }

    mram_write(wram_x, x + chunk_start, chunk_size * sizeof(float));
  }

  return 0;
}
