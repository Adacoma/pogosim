#ifndef FLASH_STATE_H
#define FLASH_STATE_H

#include <filesystem>
#include <memory>
#include <vector>

class PogobotObject;

namespace pogosim::flash_state {

/** Create a valid zero-record archive only if the input path is absent. */
bool create_empty_if_missing(const std::filesystem::path& filename);

/**
 * Restore the persistent memory of every simulated robot from an archive.
 *
 * Every supplied (category, robot ID) identity must be present. Extra archive
 * records are validated but do not create robots. A zero-record archive is a
 * fresh-state marker and leaves flash untouched. Returns false for that marker.
 * No controller or physical state is restored.
 */
bool load(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
);

/**
 * Atomically save the persistent memory of every simulated robot.
 *
 * Only the 1472 KiB user flash section and motor direction/power calibration
 * memories are stored. If input_filename is nonempty, its unused robot records
 * are preserved while supplied robots' records are replaced with their final
 * memories. A zero-record input is replaced by a full first archive. The
 * input is re-read at export, so it must stay available and unchanged during
 * the run. The temporary file is written beside the destination so input and
 * output may safely name the same archive.
 */
void save_atomic(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots,
    const std::filesystem::path& input_filename = {}
);

} // namespace pogosim::flash_state

#endif // FLASH_STATE_H
