#ifndef POGOSIM_START_DISPATCH_POGOBOT_STUB_H
#define POGOSIM_START_DISPATCH_POGOBOT_STUB_H

// Reuse API type declarations, then select the real-robot macros in pogosim.h.
// This fixture checks dispatch/category filtering, not the firmware ABI/SDK.
#include "pogosim/spogobot.h"
#undef SIMULATOR

#endif
