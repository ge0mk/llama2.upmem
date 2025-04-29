#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dpu.h"
#include "dpu_types.h"

#include "kernels/model_config.h"
#include "transformer.h"

#ifdef EMBED_KERNELS
static uint8_t attout_prog[] = {
#embed "build/attout.kernel"
};
static uint8_t cls_prog[] = {
#embed "build/cls.kernel"
};
static uint8_t ffn1_prog[] = {
#embed "build/ffn1.kernel"
};
static uint8_t ffn2_prog[] = {
#embed "build/ffn2.kernel"
};
static uint8_t mha_prog[] = {
#embed "build/mha.kernel"
};
static uint8_t qkv_prog[] = {
#embed "build/qkv.kernel"
};
static uint8_t rmsnorm_prog[] = {
#embed "build/rmsnorm.kernel"
};

#define load_dpu_kernel(dpu_set, name)                                         \
  dpu_load_from_memory(dpu_set, name##_prog, sizeof(name##_prog), nullptr)
#else

#define load_dpu_kernel(dpu_set, name)                                         \
  dpu_load(dpu_set, "build/" #name ".kernel", nullptr)
#endif

dpu_error_t dpu_log_read_all(struct dpu_set_t mha_dpus) {
  size_t i = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(mha_dpus, dpu, i) {
    dpu_error_t err = dpu_log_read(dpu, stdout);
    if (err != DPU_OK) {
      return err;
    }
  }
  (void)i;
  return DPU_OK;
}

float *forward_upmem(Transformer *transformer, int token, int pos) {
  // a few convenience variables
  Config *p = &transformer->config;
  RunState *s = &transformer->state;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul =
      p->n_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  const TransformerWeights *w = &transformer->weights;
  static float *x, *xb, *hb, *att, *q, *kc, *vc, *tkc, *tvc, *logits;
  static const int zero[2] = {0, 0};

  static struct DpuSets {
    struct dpu_set_t qkv;
    struct dpu_set_t mha;
    struct dpu_set_t cls;
    struct dpu_set_t ffn1;
    struct dpu_set_t ffn2;
    struct dpu_set_t attnout;
    struct dpu_set_t rmsnorm;
  } *dpus = nullptr;

  size_t i = 0;
  struct dpu_set_t dpu;

  if (!dpus) {
    const char *upmem_profile = getenv("UPMEM_PROFILE");

    dpus = malloc(sizeof(*dpus));
    DPU_ASSERT(dpu_alloc(VOCAB_SIZE / 16 / CLS_ROWS_PER_THREAD, upmem_profile,
                         &dpus->cls));
    DPU_ASSERT(dpu_alloc(HIDDEN_DIM / 16 / 4, upmem_profile, &dpus->ffn1));
    DPU_ASSERT(dpu_alloc(N_HEADS, upmem_profile, &dpus->mha));
    DPU_ASSERT(dpu_alloc(DIM / (QKV_TASKLETS * 2), upmem_profile, &dpus->qkv));
    DPU_ASSERT(dpu_alloc(1, upmem_profile, &dpus->rmsnorm));

    dpus->attnout = dpus->ffn2 = dpus->qkv;

    load_dpu_kernel(dpus->cls, cls);
    load_dpu_kernel(dpus->ffn1, ffn1);
    load_dpu_kernel(dpus->mha, mha);
    load_dpu_kernel(dpus->rmsnorm, rmsnorm);

    // weights don't change between layers, so we only load them once
    DPU_FOREACH(dpus->cls, dpu, i) {
      dpu_prepare_xfer(dpu, w->wcls + i * 16 * CLS_ROWS_PER_THREAD * DIM);
    }
    dpu_push_xfer(dpus->cls, DPU_XFER_TO_DPU, "wcls", 0,
                  16 * CLS_ROWS_PER_THREAD * DIM * sizeof(float),
                  DPU_XFER_DEFAULT);

    x = malloc(DIM * sizeof(float));
    xb = malloc(DIM * sizeof(float));
    hb = malloc(HIDDEN_DIM * sizeof(float));
    att = malloc(SEQ_LEN * N_HEADS * sizeof(float));
    q = malloc(DIM * sizeof(float));
    kc = malloc(N_LAYERS * SEQ_LEN * KV_DIM * sizeof(float));
    vc = malloc(N_LAYERS * SEQ_LEN * KV_DIM * sizeof(float));
    tkc = malloc(N_HEADS * SEQ_LEN * HEAD_SIZE * sizeof(float));
    tvc = malloc(N_HEADS * HEAD_SIZE * SEQ_LEN * sizeof(float));
    logits = malloc(VOCAB_SIZE * sizeof(float));
  }

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * DIM;
  memcpy(x, content_row, DIM * sizeof(*x));
  memcpy(s->x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (size_t l = 0; l < N_LAYERS; l++) {
    // key and value point to the kv cache
    int loff = l * SEQ_LEN * KV_DIM; // kv cache layer offset for convenience
    float *k = kc + loff + pos * KV_DIM;
    float *v = vc + loff + pos * KV_DIM;

    { // attention rmsnorm
      dpu_broadcast_to(dpus->rmsnorm, "w", 0, w->rms_att_weight + l * DIM,
                       DIM * sizeof(float), DPU_XFER_DEFAULT);
      dpu_broadcast_to(dpus->rmsnorm, "x", 0, x, DIM * sizeof(float),
                       DPU_XFER_DEFAULT);
      dpu_broadcast_to(dpus->rmsnorm, "data", 0, &zero, 2 * sizeof(float),
                       DPU_XFER_DEFAULT);

      dpu_launch(dpus->rmsnorm, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->rmsnorm, dpu, i) { dpu_prepare_xfer(dpu, xb); }
      dpu_push_xfer(dpus->rmsnorm, DPU_XFER_FROM_DPU, "x", 0,
                    DIM * sizeof(float), DPU_XFER_DEFAULT);
    }

    { // qkv matmuls & RoPE
      struct {
        uint32_t dpu;
        uint32_t pos;
      } data[DIM / (QKV_TASKLETS * 2)];

      for (size_t i = 0; i < DIM / (QKV_TASKLETS * 2); i++) {
        data[i].dpu = i;
        data[i].pos = pos;
      }

      load_dpu_kernel(dpus->qkv, qkv);
      dpu_broadcast_to(dpus->qkv, "x", 0, xb, DIM * sizeof(float),
                       DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, w->wq + (l * DIM * DIM) +
                                  (i * QKV_TASKLETS * 2 * DIM));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_TO_DPU, "wq", 0,
                    QKV_TASKLETS * 2 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, w->wk + (l * DIM * KV_DIM) +
                                  (i * QKV_TASKLETS * 2 * DIM));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_TO_DPU, "wk", 0,
                    QKV_TASKLETS * 2 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, w->wv + (l * DIM * KV_DIM) +
                                  (i * QKV_TASKLETS * 2 * DIM));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_TO_DPU, "wv", 0,
                    QKV_TASKLETS * 2 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) { dpu_prepare_xfer(dpu, data + i); }
      dpu_push_xfer(dpus->qkv, DPU_XFER_TO_DPU, "data", 0, sizeof(data[0]),
                    DPU_XFER_DEFAULT);

      dpu_launch(dpus->qkv, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, q + (i * QKV_TASKLETS * 2));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_FROM_DPU, "q", 0,
                    QKV_TASKLETS * 2 * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, k + (i * QKV_TASKLETS * 2));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_FROM_DPU, "k", 0,
                    QKV_TASKLETS * 2 * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->qkv, dpu, i) {
        dpu_prepare_xfer(dpu, v + (i * QKV_TASKLETS * 2));
      }
      dpu_push_xfer(dpus->qkv, DPU_XFER_FROM_DPU, "v", 0,
                    QKV_TASKLETS * 2 * sizeof(float), DPU_XFER_DEFAULT);
    }

    { // multihead attention
      struct {
        float scale;
        uint32_t pos;
      } data = {.scale = sqrtf(HEAD_SIZE), .pos = pos};

      // transpose key & value cache on cpu for sequential access on the dpus
      // q: layer x heads x head_size
      // kc: layer x seq_len x heads x head_size
      //   -> layer x heads x seq_len x head_size
      // vc: layer x seq_len x heads x head_size
      //   -> layer x heads x head_size x seq_len

      for (size_t h = 0; h < N_HEADS; h++) {
        for (size_t p = 0; p < SEQ_LEN; p++) {
          for (size_t o = 0; o < HEAD_SIZE; o++) {
            tkc[((h * SEQ_LEN) + p) * HEAD_SIZE + o] =
                kc[loff + ((p * N_HEADS) + h) * HEAD_SIZE + o];
            tvc[((h * HEAD_SIZE) + o) * SEQ_LEN + p] =
                vc[loff + ((p * N_HEADS) + h) * HEAD_SIZE + o];
          }
        }
      }

      dpu_broadcast_to(dpus->mha, "data", 0, &data, sizeof(data),
                       DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->mha, dpu, i) {
        dpu_prepare_xfer(dpu, q + i * HEAD_SIZE);
      }
      dpu_push_xfer(dpus->mha, DPU_XFER_TO_DPU, "q", 0,
                    HEAD_SIZE * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->mha, dpu, i) {
        dpu_prepare_xfer(dpu, tkc + i * SEQ_LEN * HEAD_SIZE);
      }
      dpu_push_xfer(dpus->mha, DPU_XFER_TO_DPU, "kc", 0,
                    HEAD_SIZE * SEQ_LEN * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->mha, dpu, i) {
        dpu_prepare_xfer(dpu, tvc + i * HEAD_SIZE * SEQ_LEN);
      }
      dpu_push_xfer(dpus->mha, DPU_XFER_TO_DPU, "vc", 0,
                    HEAD_SIZE * SEQ_LEN * sizeof(float), DPU_XFER_DEFAULT);

      dpu_launch(dpus->mha, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->mha, dpu, i) {
        dpu_prepare_xfer(dpu, xb + i * HEAD_SIZE);
      }
      dpu_push_xfer(dpus->mha, DPU_XFER_FROM_DPU, "x", 0,
                    HEAD_SIZE * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->mha, dpu, i) {
        dpu_prepare_xfer(dpu, att + i * SEQ_LEN);
      }
      dpu_push_xfer(dpus->mha, DPU_XFER_FROM_DPU, "att", 0,
                    SEQ_LEN * sizeof(float), DPU_XFER_DEFAULT);
    }

    { // attention output
      load_dpu_kernel(dpus->attnout, attout);

      DPU_FOREACH(dpus->attnout, dpu, i) { dpu_prepare_xfer(dpu, x + i * 16); }
      dpu_push_xfer(dpus->attnout, DPU_XFER_TO_DPU, "x", 0, 16 * sizeof(float),
                    DPU_XFER_DEFAULT);
      dpu_broadcast_to(dpus->attnout, "xb", 0, xb, DIM * sizeof(float),
                       DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->attnout, dpu, i) {
        dpu_prepare_xfer(dpu, w->wo + l * DIM * DIM + i * 16 * DIM);
      }
      dpu_push_xfer(dpus->attnout, DPU_XFER_TO_DPU, "wo", 0,
                    16 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      dpu_launch(dpus->attnout, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->attnout, dpu, i) { dpu_prepare_xfer(dpu, x + i * 16); }
      dpu_push_xfer(dpus->attnout, DPU_XFER_FROM_DPU, "x", 0,
                    16 * sizeof(float), DPU_XFER_DEFAULT);
    }

    { // ffn rmsnorm
      dpu_broadcast_to(dpus->rmsnorm, "w", 0, w->rms_ffn_weight + l * DIM,
                       DIM * sizeof(float), DPU_XFER_DEFAULT);
      dpu_broadcast_to(dpus->rmsnorm, "x", 0, x, DIM * sizeof(float),
                       DPU_XFER_DEFAULT);
      dpu_broadcast_to(dpus->rmsnorm, "data", 0, &zero, 2 * sizeof(float),
                       DPU_XFER_DEFAULT);

      dpu_launch(dpus->rmsnorm, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->rmsnorm, dpu, i) { dpu_prepare_xfer(dpu, xb); }
      dpu_push_xfer(dpus->rmsnorm, DPU_XFER_FROM_DPU, "x", 0,
                    DIM * sizeof(float), DPU_XFER_DEFAULT);
    }

    { // ffn
      DPU_FOREACH(dpus->ffn1, dpu, i) {
        dpu_prepare_xfer(dpu, w->w1 + l * DIM * HIDDEN_DIM + i * 4 * 16 * DIM);
      }
      dpu_push_xfer(dpus->ffn1, DPU_XFER_TO_DPU, "w1", 0,
                    4 * 16 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->ffn1, dpu, i) {
        dpu_prepare_xfer(dpu, w->w3 + l * DIM * HIDDEN_DIM + i * 4 * 16 * DIM);
      }
      dpu_push_xfer(dpus->ffn1, DPU_XFER_TO_DPU, "w3", 0,
                    4 * 16 * DIM * sizeof(float), DPU_XFER_DEFAULT);

      dpu_broadcast_to(dpus->ffn1, "xb", 0, xb, DIM * sizeof(float),
                       DPU_XFER_DEFAULT);

      dpu_launch(dpus->ffn1, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->ffn1, dpu, i) {
        dpu_prepare_xfer(dpu, hb + i * 4 * 16);
      }
      dpu_push_xfer(dpus->ffn1, DPU_XFER_FROM_DPU, "hb", 0,
                    4 * 16 * sizeof(float), DPU_XFER_DEFAULT);

      load_dpu_kernel(dpus->ffn2, ffn2);

      dpu_broadcast_to(dpus->ffn2, "hb", 0, hb, HIDDEN_DIM * sizeof(float),
                       DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->ffn2, dpu, i) {
        dpu_prepare_xfer(dpu,
                         w->w2 + l * DIM * HIDDEN_DIM + i * 16 * HIDDEN_DIM);
      }
      dpu_push_xfer(dpus->ffn2, DPU_XFER_TO_DPU, "w2", 0,
                    16 * HIDDEN_DIM * sizeof(float), DPU_XFER_DEFAULT);

      DPU_FOREACH(dpus->ffn2, dpu, i) { dpu_prepare_xfer(dpu, x + i * 16); }
      dpu_push_xfer(dpus->ffn2, DPU_XFER_TO_DPU, "x", 0, 16 * sizeof(float),
                    DPU_XFER_DEFAULT);

      dpu_launch(dpus->ffn2, DPU_SYNCHRONOUS);

      DPU_FOREACH(dpus->ffn2, dpu, i) { dpu_prepare_xfer(dpu, x + i * 16); }
      dpu_push_xfer(dpus->ffn2, DPU_XFER_FROM_DPU, "x", 0, 16 * sizeof(float),
                    DPU_XFER_DEFAULT);
    }
  }

  { // final rmsnorm
    dpu_broadcast_to(dpus->rmsnorm, "w", 0, w->rms_final_weight,
                     DIM * sizeof(float), DPU_XFER_DEFAULT);
    dpu_broadcast_to(dpus->rmsnorm, "x", 0, x, DIM * sizeof(float),
                     DPU_XFER_DEFAULT);
    dpu_broadcast_to(dpus->rmsnorm, "data", 0, &zero, 2 * sizeof(float),
                     DPU_XFER_DEFAULT);

    dpu_launch(dpus->rmsnorm, DPU_SYNCHRONOUS);

    DPU_FOREACH(dpus->rmsnorm, dpu, i) { dpu_prepare_xfer(dpu, x); }
    dpu_push_xfer(dpus->rmsnorm, DPU_XFER_FROM_DPU, "x", 0, DIM * sizeof(float),
                  DPU_XFER_DEFAULT);
  }

  { // classifier into logits
    // 20 dpus, 16 tasklets -> 320 threads -> 100 rows per thread
    dpu_broadcast_to(dpus->cls, "x", 0, x, DIM * sizeof(float),
                     DPU_XFER_DEFAULT);

    dpu_launch(dpus->cls, DPU_SYNCHRONOUS);

    DPU_FOREACH(dpus->cls, dpu, i) {
      dpu_prepare_xfer(dpu, logits + i * 16 * CLS_ROWS_PER_THREAD);
    }
    dpu_push_xfer(dpus->cls, DPU_XFER_FROM_DPU, "logits", 0,
                  16 * CLS_ROWS_PER_THREAD * sizeof(float), DPU_XFER_DEFAULT);
  }

  return logits;
}

void mha_big_test(int pos) {
  static float *q = nullptr;
  static float *kc = nullptr;
  static float *vc = nullptr;
  static float *xb = nullptr;

  static struct dpu_set_t dpu_set;
  static bool first_run = true;

  static const size_t n_heads = 8;
  static const size_t head_size = 4096;
  static const size_t seq_len = 256;

  if (first_run) {
    first_run = false;

    q = malloc(n_heads * head_size * sizeof(float));
    kc = malloc(n_heads * head_size * seq_len * sizeof(float));
    vc = malloc(n_heads * head_size * seq_len * sizeof(float));
    xb = malloc(n_heads * head_size * sizeof(float));

    dpu_alloc(n_heads, getenv("UPMEM_PROFILE"), &dpu_set);
    dpu_load(dpu_set, "build/mha_big.kernel", nullptr);
  }

  size_t i = 0;
  struct dpu_set_t dpu;

  struct {
    float scale;
    uint32_t pos;
  } data = {.scale = sqrtf(head_size), .pos = pos};

  dpu_broadcast_to(dpu_set, "data", 0, &data, sizeof(data), DPU_XFER_DEFAULT);

  DPU_FOREACH(dpu_set, dpu, i) { dpu_prepare_xfer(dpu, q + i * head_size); }
  dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "q", 0, head_size * sizeof(float),
                DPU_XFER_DEFAULT);

  DPU_FOREACH(dpu_set, dpu, i) {
    dpu_prepare_xfer(dpu, kc + i * seq_len * head_size);
  }
  dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "kc", 0,
                head_size * seq_len * sizeof(float), DPU_XFER_DEFAULT);

  DPU_FOREACH(dpu_set, dpu, i) {
    dpu_prepare_xfer(dpu, vc + i * head_size * seq_len);
  }
  dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "vc", 0,
                head_size * seq_len * sizeof(float), DPU_XFER_DEFAULT);

  dpu_launch(dpu_set, DPU_SYNCHRONOUS);

  DPU_FOREACH(dpu_set, dpu, i) { dpu_prepare_xfer(dpu, xb + i * head_size); }
  dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "x", 0, head_size * sizeof(float),
                DPU_XFER_DEFAULT);
}
