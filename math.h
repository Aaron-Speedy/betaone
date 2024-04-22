#ifndef MATH_H
#define MATH_H

#include <stdlib.h>
#include <assert.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

typedef struct {
  int x, y;
} Vec2I;

typedef struct {
  float *buf;
  int w, h;
} Matrix;

#define m_at(m, i, j) (m)->buf[(i) + (j) * (m)->w]
void matrix_init(Arena *al, Matrix *m);
void matrix_print(Matrix m);
void matrix_randomize(Matrix *m, float min, float max);
Matrix kernel_apply(Arena *al, Matrix *in, Matrix *k, int hs, int vs);
Matrix matrix_multiply(Arena *al, Matrix *a, Matrix *b);

float randf(float min, float max);
int randi(int min, int max);
int sign(int x);
void clampf(float *x, float min, float max);
void clampi(int *x, int min, int max);
int in_bounds(int x, int max);
int vec2i_at(Vec2I vec, int index);

#ifdef MATH_IMPL
#define MATH_IMPL

void matrix_init(Arena *al, Matrix *m) {
  m->buf = arena_alloc(al, m->w * m->h * sizeof(float));
  memset(m->buf, 0.0, m->w * m->h * sizeof(float));
}

void matrix_print(Matrix m) {
  for (int j = 0; j < m.h; j++) {
    for (int i = 0; i < m.w; i++) {
      printf("%.3f ", m_at(&m, i, j));
    }
    printf("\n");
  }
}

void matrix_randomize(Matrix *m, float min, float max) {
  for (int i = 0; i < m->w; i++) {
    for (int j = 0; j < m->h; j++) {
      m_at(m, i, j) = randf(min, max);
    }
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

float randf(float min, float max) {
  float scale = rand() / (float) RAND_MAX;
  return min + scale * ( max - min );
}

int randi(int min, int max) {
  return (rand() % (max - min)) + min + 1;
}

int sign(int x) {
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

void clampf(float *x, float min, float max) {
  if (*x < min) *x = min;
  else if (*x > max) *x = max;
}

void clampi(int *x, int min, int max) {
  if (*x < min) *x = min;
  else if (*x > max) *x = max;
}

int in_bounds(int x, int max) {
  if (x < 0) return 0;
  if (x >= max) return 0;
  return 1;
}

int vec2i_at(Vec2I vec, int index) {
  switch (index) {
  case 0:
    return vec.x;
  case 1:
    return vec.y;
  default:
    assert(!"Out of bounds");
  }
}

#endif

#endif
