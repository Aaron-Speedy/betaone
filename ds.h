#ifndef DS_H
#define DS_H

#include <stdlib.h>
#include <assert.h>

typedef struct {
  char *buf;
  size_t len;
  size_t pos;
} Arena;

void arena_init(Arena *al);
void *arena_alloc(Arena *al, size_t len);

#define List_F(T) T *items; size_t cap, count
#define List(T) struct { List_F(T); }

#define da_init(xs) \
do { \
  (xs)->items = malloc((xs)->cap * sizeof((xs)->items[0])); \
} while (0);

#define da_init_ar(arena, xs) \
do { \
  (xs)->items = arena_malloc((arena), (xs)->cap * sizeof((xs)->items[0])); \
} while (0);

#define da_push(xs, x) \
  do { \
    if ((xs)->count >= (xs)->cap) { \
      assert((xs)->cap != 0); \
      (xs)->cap *= 2; \
      (xs)->items = realloc((xs)->items, (xs)->cap*sizeof(*(xs)->items)); \
    } \
\
    (xs)->items[(xs)->count++] = (x); \
  } while (0)

#define da_push_ar(arena, xs, x) \
do { \
  if ((xs)->count >= (xs)->cap) { \
    assert((xs)->cap != 0); \
    (xs)->cap *= 2; \
    (xs)->items = arena_alloc((arena), (xs)->cap)); \
  } \
\
  (xs)->items[(xs)->count++] = (x); \
}

#define da_pop(xs) \
  do { \
    assert ((xs)->count > 0); \
    (xs)->count -= 1; \
  } while (0)

#define da_last(xs) (xs)->items[(xs)->count - 1]

// count represents width of used region
#define Pool_F(T) T *items; int cap, count; List(int) free_list
#define Pool(T) struct { Pool_F(T); }

#define pool_init(xs) \
  do { \
    (xs)->count = 0; \
    (xs)->cap = 256; \
    (xs)->items = malloc((xs)->cap * sizeof((xs)->items[0])); \
    (xs)->free_list.count = 0; \
    (xs)->free_list.cap = 256; \
    (xs)->free_list.items = malloc((xs)->free_list.cap * sizeof((xs)->free_list.items[0])); \
  } while (0);

#define pool_add(xs, x) \
  do { \
    if ((xs)->free_list.count == 0) da_push(xs, x); \
    else { \
      (xs)->items[da_last(&((xs)->free_list))] = x; \
      da_pop(&((xs)->free_list)); \
    } \
  } while(0);

#define pool_del(xs, i) da_push(&((xs)->free_list), i)

#ifdef DS_IMPL
#define DS_IMPL

void arena_init(Arena *al) {
  al->buf = malloc(al->len);
}

void *arena_alloc(Arena *al, size_t len) {
  assert(al->pos + len <= al->len && "Not enough memory in arena");

  void *ptr = &al->buf[al->pos];
  al->pos += len;

  return ptr;
}

#endif

#endif
