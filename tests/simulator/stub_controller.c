#include "pogobase.h"

/* Supply the controller symbols required by the archive, but never run a
 * controller in model/unit tests. CLI integration uses a separate real one. */
typedef struct { unsigned reserved; } USERDATA;
REGISTER_USERDATA(USERDATA);

int main(void) { return 0; }
