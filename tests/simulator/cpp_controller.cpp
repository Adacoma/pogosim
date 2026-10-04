#include "pogobase.h"
#include <type_traits>

// C++ controllers retain the existing explicit C-linkage entrypoint. Include
// the unchanged legacy C fixture as C++ to exercise both userdata macros and
// callback declarations without a manual extern "C" wrapper around them.
extern "C" int robot_main(void);
#include "legacy_controller.c"

// Redeclarations after the public macros/headers must agree on C linkage.
// This catches missing extern "C" even on Unix ABIs where global variable
// names happen to match despite inconsistent C/C++ language linkage.
extern "C" {
extern size_t UserdataSize;
extern USERDATA *mydata;
extern void (*callback_create_data_schema)(void);
extern void (*callback_export_data)(void);
extern void (*callback_global_setup)(void);
extern void (*callback_global_step)(void);
extern void (*callback_robot_end)(void);
extern void (*callback_robot_click)(void);
}

static_assert(std::is_same_v<decltype(UserdataSize), size_t>,
              "Controller userdata size must retain its size_t type");
