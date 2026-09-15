# Flash-state import/export test

This example checks persistent memory across two separate starts, both in
Pogosim and on real Pogobots. It covers the 64 KiB user flash section, motor
direction memory, and motor power memory. The simulator test additionally
checks robot identity mapping, callback ordering, and using one pathname for
both import and atomic export.

## Simulator

Build the example using the standard Pogosim workflow, then run from the
repository root:

```console
./examples/test_flash_state/test_flash_state \
  -c examples/test_flash_state/conf/export.yaml -g

./examples/test_flash_state/test_flash_state \
  -c examples/test_flash_state/conf/import.yaml -g
```

The first invocation creates `frames/test_flash_state.pgflash`. The second must
print one successful import line for each robot and exit with status zero:

```text
FLASH_STATE IMPORT: robot 0 OK
FLASH_STATE IMPORT: robot 1 OK
```

Run the export phase again whenever a fresh test archive is wanted. The
archive is deliberately ignored by Git through the repository's `frames/`
rule.

## Real robots

Build and upload the example with the normal Pogobot SDK workflow. The same
firmware performs both phases automatically:

1. On the first boot, when no test marker is present, it erases the complete
   64 KiB user-writable flash section, writes the fixture, snapshots the
   existing motor calibration memories, and turns the LED blue.
2. Power-cycle the robot without reflashing it. The next boot verifies the
   flash fixture and checks that the motor calibration memories still match the
   snapshot. Success turns the LED green; failure turns it red.

The test does not replace real motor calibration with synthetic values.
However, its first phase deliberately erases all existing user-writable flash,
so it must only be run when losing that data is acceptable. Erase the test
section or upload firmware that does so before repeating the first phase.
