# Persistent robot flash state

Pogosim can optionally restore and save the persistent memory of every
simulated robot. This supports experiments split across multiple simulator
invocations without checkpointing positions, controller RAM, timers, or other
transient simulation state.

## Configuration

Configure either operation independently:

```yaml
flash_state:
  input_file: "checkpoints/previous.pgflash"
  output_file: "checkpoints/final.pgflash"
```

`input_file` is loaded after all robots have been created but before
`robot_main()` and `user_init()` run. `output_file` is written after all
per-robot end-of-experiment callbacks have completed. Either key may be
omitted. If both are omitted, Pogosim neither reads nor stores flash content.

The input and output names may refer to the same file. Pogosim writes a
temporary file beside the destination and atomically replaces the old archive
only after the new archive is complete.
Fatal simulator errors do not replace the previous archive; export occurs only
after the main loop and robot end callbacks complete normally.

Relative paths follow the simulator's existing output-file convention and are
interpreted from its working directory.

## Stored state

Each robot record contains only persistent state represented by the simulated
Pogobot API:

- the 64 KiB user-writable flash section;
- the three motor-direction calibration values;
- the three motor-power calibration values.

Position, orientation, controller `USERDATA`, clocks, messages, sensor state,
and simulator state are deliberately excluded. When no input archive is given,
the user flash remains uninitialized and indeterminate, as requested for a
fresh simulated robot; the motor calibration memories retain their simulator
defaults.

The archive is a versioned binary format. Records are associated with robots
by `(category, robot_id)` and protected by per-record checksums. Loading fails
before controller initialization if the format, robot count, identity set,
flash size, checksum, or file length does not match. Consequently, merely using
the same robot count is insufficient if categories or ID assignment changed.

The controller defines the meaning and layout of its flash bytes. Pogosim
cannot detect an archive produced by an incompatible controller, so controller
flash layouts should contain their own application-level version or magic
value when they may evolve.

## Pogobatch behavior

Pogobatch leaves `input_file` unchanged so all tasks can start from the same
checkpoint. It redirects each task's `output_file` into that task's artifact
directory, preventing concurrent simulations from overwriting one another.
These task-level archives are not merged into the Feather result. Local
campaigns remove successful task directories by default, so use Pogobatch's
`--keep-temp` option when the individual final flash archives must be retained.
