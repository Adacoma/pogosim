// Test persistent flash-state export and import across simulator invocations.

#include "pogobase.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    // A non-empty USERDATA type is required, but transient state is not part of
    // this persistence test and must never be written to the flash archive.
    uint8_t unused;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

enum {
    FLASH_STATE_MODE_HARDWARE_AUTO = 0,
    FLASH_STATE_MODE_EXPORT = 1,
    FLASH_STATE_MODE_IMPORT = 2,
    TEST_FLASH_PAGE = 7,
    TEST_METADATA_PAGE = 8,
};

// This value is overridden from YAML by global_setup() in simulation. On real
// robots callbacks are disabled, so the hardware-safe automatic mode remains.
static uint8_t flash_state_test_mode = FLASH_STATE_MODE_HARDWARE_AUTO;

static const uint8_t fixture_magic[8] = {
    'P', 'G', 'F', 'L', 'T', 'E', 'S', 'T'
};

static uint8_t expected_flash_byte(uint16_t robot_id, uint16_t offset) {
    return (uint8_t)((0x5au + 17u * robot_id + 7u * offset) & 0xffu);
}

#ifdef SIMULATOR
static void expected_motor_state(
    uint16_t robot_id,
    uint8_t directions[3],
    uint16_t powers[3]
) {
    // Direction values remain valid motor directions while still depending on
    // robot identity, so assigning one robot's archive record to another fails.
    directions[0] = (uint8_t)(robot_id & 1u);
    directions[1] = (uint8_t)((robot_id + 1u) & 1u);
    directions[2] = (uint8_t)(robot_id & 1u);

    powers[0] = (uint16_t)(100u + robot_id);
    powers[1] = (uint16_t)(200u + robot_id);
    powers[2] = (uint16_t)(300u + robot_id);
}
#endif

static void put_u16_le(uint8_t destination[2], uint16_t value) {
    destination[0] = (uint8_t)(value & 0xffu);
    destination[1] = (uint8_t)(value >> 8);
}

static uint16_t get_u16_le(const uint8_t source[2]) {
    return (uint16_t)source[0] | ((uint16_t)source[1] << 8);
}

static void build_metadata_page(
    uint8_t page[256],
    uint16_t robot_id,
    const uint8_t directions[3],
    const uint16_t powers[3]
) {
    memset(page, 0xff, 256);
    memcpy(page, fixture_magic, sizeof(fixture_magic));
    put_u16_le(&page[8], robot_id);
    memcpy(&page[10], directions, 3);
    for (uint8_t i = 0; i < 3; ++i) {
        put_u16_le(&page[13 + 2 * i], powers[i]);
    }
}

#ifndef SIMULATOR
static bool fixture_is_present(void) {
    uint8_t metadata[256];
    read_page_flash(TEST_METADATA_PAGE, (char *)metadata);
    return memcmp(metadata, fixture_magic, sizeof(fixture_magic)) == 0;
}
#endif

// Called at the end of the export run. This specifically verifies that the
// simulator writes its archive after per-robot end callbacks have completed.
static void prepare_persistent_state(void) {
    const uint16_t robot_id = pogobot_helper_getid();
    uint8_t page[256];
    uint8_t metadata[256];
    uint8_t directions[3];
    uint16_t powers[3];

#ifdef SIMULATOR
    // Synthetic per-ID motor values ensure that the simulator archive really
    // restores motor memories instead of retaining their constructor defaults.
    expected_motor_state(robot_id, directions, powers);
#else
    // Preserve real calibration: snapshot and rewrite its existing values
    // rather than substituting synthetic test calibration.
    pogobot_motor_dir_mem_get(directions);
    pogobot_motor_power_mem_get(powers);
#endif

    erase_write_section_flash();
    for (uint16_t i = 0; i < 256; ++i) {
        page[i] = expected_flash_byte(robot_id, i);
    }
    write_page_flash(TEST_FLASH_PAGE, page);

    build_metadata_page(metadata, robot_id, directions, powers);
    write_page_flash(TEST_METADATA_PAGE, metadata);

#ifdef SIMULATOR
    pogobot_motor_dir_mem_set(directions);
    pogobot_motor_power_mem_set(powers);
#endif

    printf("FLASH_STATE PREPARE: robot %u prepared\n", robot_id);
}

// Called during user_init() in the import run. Any success therefore proves
// that restoration happened before the controller initialization callback.
static void verify_persistent_state(void) {
    const uint16_t robot_id = pogobot_helper_getid();
    uint8_t page[256];
    uint8_t metadata[256];
    uint8_t directions[3];
    uint8_t expected_directions[3];
    uint16_t powers[3];
    uint16_t expected_powers[3];
    unsigned int errors = 0;

    read_page_flash(TEST_FLASH_PAGE, (char *)page);
    for (uint16_t i = 0; i < 256; ++i) {
        errors += page[i] != expected_flash_byte(robot_id, i);
    }

    read_page_flash(TEST_METADATA_PAGE, (char *)metadata);
    errors += memcmp(metadata, fixture_magic, sizeof(fixture_magic)) != 0;
    errors += get_u16_le(&metadata[8]) != robot_id;
    memcpy(expected_directions, &metadata[10], 3);
    for (uint8_t i = 0; i < 3; ++i) {
        expected_powers[i] = get_u16_le(&metadata[13 + 2 * i]);
    }

    if (pogobot_motor_dir_mem_get(directions) != 0) {
        ++errors;
    }
    if (pogobot_motor_power_mem_get(powers) != 0) {
        ++errors;
    }
    for (uint8_t i = 0; i < 3; ++i) {
        errors += directions[i] != expected_directions[i];
        errors += powers[i] != expected_powers[i];
    }

    if (errors != 0) {
        printf(
            "FLASH_STATE IMPORT: robot %u FAILED (%u mismatches)\n",
            robot_id,
            errors
        );
        pogobot_led_setColor(255, 0, 0);
#ifdef SIMULATOR
        // A failing process status makes the example usable as a smoke test.
        exit(EXIT_FAILURE);
#endif
        return;
    }

    pogobot_led_setColor(0, 255, 0);
    printf("FLASH_STATE IMPORT: robot %u OK\n", robot_id);
}

#ifdef SIMULATOR
// Called once by the simulator after every robot has registered its callbacks
// and before user_init() runs. Configuration access belongs in this callback.
static void global_setup(void) {
    init_from_configuration(flash_state_test_mode);
}

// The simulator exports its archive after this callback returns.
static void robot_end(void) {
    if (flash_state_test_mode == FLASH_STATE_MODE_EXPORT) {
        prepare_persistent_state();
    }
}
#endif

void user_init(void) {
#ifdef SIMULATOR

    if (flash_state_test_mode == FLASH_STATE_MODE_IMPORT) {
        verify_persistent_state();
    } else if (flash_state_test_mode != FLASH_STATE_MODE_EXPORT) {
        printf(
            "FLASH_STATE: invalid flash_state_test_mode %u; expected 1 or 2\n",
            flash_state_test_mode
        );
        exit(EXIT_FAILURE);
    }
#else
    // Real flash persists without an archive. The same firmware prepares a
    // fixture on its first boot and verifies it on subsequent boots.
    if (fixture_is_present()) {
        verify_persistent_state();
    } else {
        printf(
            "FLASH_STATE: no fixture found; erasing the 64 KiB user section\n"
        );
        prepare_persistent_state();
        pogobot_led_setColor(0, 0, 255);
        printf("FLASH_STATE: power-cycle the robot to verify persistence\n");
    }
#endif
}

void user_step(void) {
    // The configurations use simulation_time: 0; no controller step is needed.
}

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);

    // These callbacks compile away on hardware. global_setup owns simulator
    // configuration, following the same lifecycle as the other examples.
    SET_CALLBACK(callback_global_setup, global_setup);
    SET_CALLBACK(callback_robot_end, robot_end);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
