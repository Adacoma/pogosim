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
  create_if_missing: true  # default; set false to require an existing input
```

`input_file` is loaded after all robots have been created but before
`robot_main()` and `user_init()` run. `output_file` is written after all
per-robot end-of-experiment callbacks have completed. Either key may be
omitted. If both are omitted, Pogosim neither reads nor stores flash content.

If `input_file` is specified but absent, Pogosim creates a valid zero-record
archive by default. This marker contains no robot flash or motor memories:
the first run starts with its usual indeterminate user flash and default motor
calibration. An `output_file` is needed to save the final memories; when it is
the same path as `input_file`, the completed run replaces the empty marker with
a full archive. Without `output_file`, the input remains an empty marker.
Set `create_if_missing: false` to fail when the input path does not exist.
Existing malformed archives are never treated as empty and still fail.

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

- the full v3 user-writable flash section: 5,888 pages of 256 bytes
  (1,507,328 bytes, or 1,472 KiB);
- the three motor-direction calibration values;
- the three motor-power calibration values.

Position, orientation, controller `USERDATA`, clocks, messages, sensor state,
and simulator state are deliberately excluded. When no input archive is given,
the user flash remains uninitialized and indeterminate, as requested for a
fresh simulated robot; the motor calibration memories retain their simulator
defaults.

The archive is a versioned binary format. Records are associated with robots
by `(category, robot_id)` and protected by per-record checksums. Except for
the zero-record fresh-state marker, every robot in the new simulation must
have a matching archive record, but the archive may contain more robots.
Extra records are fully checked and ignored during the simulation; they do not
create robots. A missing identity, too few records, bad format, wrong flash
size, bad checksum, duplicate identity, or incorrect file length still causes
loading to fail before controller initialization.
Matching counts alone never substitutes for matching identities.
Archives containing the former 64 KiB flash region are rejected as the wrong
size; they cannot restore the complete v3 region. Each robot record is now
about 23 times larger, and simulated flash uses 1,472 KiB of RAM per robot.

When `input_file` is set, export retains records for robots absent from the
current simulation and replaces records for simulated robots with their final
flash and motor calibration memories. This also works when `input_file` and
`output_file` are the same path. For example, a five-robot archive can be used
by a two-robot run and still be imported later by a five-robot run. With no
input archive, export contains only the robots in the current simulation. To
preserve the unused records, export re-reads the input archive; keep it
available and do not modify it concurrently during the simulation.

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
