#include "pogobase.h"
#undef main

/* Test the installed public macros without SDL, a simulator, or heap state.
 * Compile the same fixture as C/C++, and both MSVC preprocessor modes. */
static unsigned registrations;
static void (*registered_init)(void);
static void (*registered_step)(void);
static const char *registered_category;
static unsigned category_reads;

static void fixture_init(void) {}
static void fixture_step(void) {}

#ifdef __cplusplus
extern "C" {
#endif
#ifdef SIMULATOR
void _pogobot_start(void (*initialize)(void), void (*step)(void), const char *category) {
    registered_category = category;
#else
void _pogobot_start(void (*initialize)(void), void (*step)(void)) {
    registered_category = get_current_robot_category();
#endif
    ++registrations;
    registered_init = initialize;
    registered_step = step;
}
#ifdef __cplusplus
}
#endif

static void require(int condition, const char *message) {
    /* Release builds must also reject a macro that silently becomes a no-op. */
    if (!condition) {
        fprintf(stderr, "Start dispatch: %s\n", message);
        exit(23);
    }
}

static const char *fixture_category(void) {
    ++category_reads;
    return "robots";
}

int main(void) {
    pogobot_start(fixture_init, fixture_step);
    require(registrations == 1, "Two-argument call did not register controllers");
    require(registered_init == fixture_init && registered_step == fixture_step,
            "Two-argument call changed controller pointers");
    require(strcmp(registered_category, "robots") == 0, "Default category changed");

    pogobot_start(fixture_init, fixture_step, fixture_category());
    require(registrations == 2, "Three-argument call did not register controllers");
    require(registered_init == fixture_init && registered_step == fixture_step,
            "Three-argument call changed controller pointers");
    require(category_reads == 1, "Category expression was not evaluated exactly once");

    /* Check forwarded argument packs as well as direct calls. */
#define FIXTURE_DEFAULT_ARGS fixture_init, fixture_step
#define FIXTURE_EXPLICIT_ARGS fixture_init, fixture_step, "robots"
    pogobot_start(FIXTURE_DEFAULT_ARGS);
    pogobot_start(FIXTURE_EXPLICIT_ARGS);
    require(registrations == 4, "Forwarded argument packs did not register controllers");

    pogobot_start(fixture_init, fixture_step, "robots2");
#ifdef SIMULATOR
    require(registrations == 5 && strcmp(registered_category, "robots2") == 0,
            "Explicit simulator category was not forwarded");
#else
    require(registrations == 4, "Hardware category filter accepted another category");
#endif
    return 0;
}
