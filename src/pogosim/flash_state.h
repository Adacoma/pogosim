#ifndef FLASH_STATE_H
#define FLASH_STATE_H

#include <filesystem>
#include <memory>
#include <vector>

class PogobotObject;

namespace pogosim::flash_state {

/**
 * Restore the persistent memory of every simulated robot from an archive.
 *
 * The archive must contain exactly the same (category, robot ID) identities as
 * the supplied robot collection. No controller or physical state is restored.
 */
void load(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
);

/**
 * Atomically save the persistent memory of every simulated robot.
 *
 * Only the 64 KiB user flash section and the motor direction/power calibration
 * memories are stored. The temporary file is written beside the destination so
 * that input and output may safely name the same archive.
 */
void save_atomic(
    const std::filesystem::path& filename,
    const std::vector<std::shared_ptr<PogobotObject>>& robots
);

} // namespace pogosim::flash_state

#endif // FLASH_STATE_H

