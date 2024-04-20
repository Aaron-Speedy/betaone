#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DS_IMPL
#include "ds.h"

#define MATH_IMPL
#include "math.h"

typedef struct {
  float *buf;
  int w, h;
} Matrix;

#define m_at(m, i, j) (m)->buf[(i) + (j) * (m)->w]

typedef struct {
  enum {
    ACT_RELU,
    ACT_TANH,
  } act;
} NN;

void matrix_init(Arena *al, Matrix *m) {
  m->buf = arena_alloc(al, m->w * m->h * sizeof(float));
  memset(m->buf, 0.0, m->w * m->h * sizeof(float));
}

void matrix_print(Matrix *m) {
  for (int j = 0; j < m->h; j++) {
    for (int i = 0; i < m->w; i++) {
      printf("%.0f ", m_at(m, i, j));
    }
    printf("\n");
  }
}

Matrix kernel_apply(Arena *al, Matrix *in, Matrix *k, int hs, int vs) {
  assert(k->w % 2);
  assert(k->h % 2);

  Matrix out = {
    .w = in->w / hs,
    .h = in->h / vs,
  };
  matrix_init(al, &out);

  for (int i = 0; i <= in->w + hs; i += hs) {
    for (int j = 0; j < in->h + vs; j += vs) {
      for (int x = 0; x < k->w; x++) {
        for (int y = 0; y < k->h; y++) {
          int ox = i + x - k->w/2, 
              oy = j + y - k->h/2;
          float v = in_bounds(ox, in->w) && in_bounds(oy, in->h) ?
                    m_at(in, ox, oy) : 0;
          m_at(&out, i, j) += v * m_at(k, x, y);
        }
      }
    }
  }

  return out;
}

Matrix matrix_multiply(Arena *al, Matrix *a, Matrix *b) {
  assert(a->w == b->h);

  Matrix out = { .w = b->w, .h = a->h };
  matrix_init(al, &out);

  for (int i = 0; i < out.w; i++) {
    for (int j = 0; j < out.h; j++) {
      for (int k = 0; k < a->h; k++) {
        m_at(&out, i, j) += m_at(b, i, k) * m_at(a, k, j);
      }
    }
  }

  return out;
}

int main() {
  srand(time(0));

  Arena al = { .len = 10000, };
  arena_init(&al);

  Matrix a = { .w = 2, .h = 2, };
  matrix_init(&al, &a);

  for (int i = 0; i < a.w; i++) {
    for (int j = 0; j < a.h; j++) {
      m_at(&a, i, j) = randi(0, 10);
    }
  }

  Matrix b = { .w = 2, .h = 2, };
  matrix_init(&al, &b);

  for (int i = 0; i < b.w; i++) {
    for (int j = 0; j < b.h; j++) {
      m_at(&b, i, j) = randi(0, 10);
    }
  }

  Matrix out = matrix_multiply(&al, &a, &b);

  matrix_print(&a);
  printf("\n");
  matrix_print(&b);
  printf("\n");
  matrix_print(&out);

  return 0;
}
