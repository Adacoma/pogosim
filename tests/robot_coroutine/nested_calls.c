#include <stdint.h>

// Compile this as C: suspension must preserve ordinary, nested C stack frames.
static int inner(void (*sleep_until)(void *, uint64_t), void *context, int value) {
    int saved = value + 7;
    sleep_until(context, 5000);
    return saved + value;
}

int coroutine_nested_c(void (*sleep_until)(void *, uint64_t), void *context) {
    int saved = 35;
    return saved + inner(sleep_until, context, saved);
}
