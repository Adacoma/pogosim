/* 
 * This file is modified by hand when a new release is prepared.
*/
#ifndef POGOSIM_VERSION_H
#define POGOSIM_VERSION_H
/** @file version.h
 *  Canonical, single-source definition of the simulator’s semantic version.
 *
 *  All helpers resolve at preprocessing time; they add zero run-time cost.
 */

#define POGOLIB_RELEASE_VERSION "v2.6"


/* ────────── 1. Canonical numeric components ────────── */
#define POGOSIM_VERSION_MAJOR   0
#define POGOSIM_VERSION_MINOR   10
#define POGOSIM_VERSION_PATCH   1

/* ────── 2. Internal helpers for stringification (private) ────── */
#define POGOSIM_VERSION__STR_IMPL(x) #x
#define POGOSIM_VERSION__STR(x)      POGOSIM_VERSION__STR_IMPL(x)

/* ─── 3. Human-readable “vX.Y.Z” ─── */
#define POGOSIM_VERSION  \
        "v" POGOSIM_VERSION__STR(POGOSIM_VERSION_MAJOR) "." \
           POGOSIM_VERSION__STR(POGOSIM_VERSION_MINOR) "." \
           POGOSIM_VERSION__STR(POGOSIM_VERSION_PATCH)

/* Alias */
#define POGOSIM_VERSION_STRING  POGOSIM_VERSION

/* ───── 4. Numeric helpers inspired from SDL ───── */
#define POGOSIM_VERSION_ENCODE(x, y, z)   ((x) * 1000000 + (y) * 1000 + (z))

#define POGOSIM_VERSION_COMPILED \
        POGOSIM_VERSION_ENCODE(POGOSIM_VERSION_MAJOR, \
                               POGOSIM_VERSION_MINOR, \
                               POGOSIM_VERSION_PATCH)

#define POGOSIM_VERSION_ATLEAST(x, y, z) \
        (POGOSIM_VERSION_COMPILED >= POGOSIM_VERSION_ENCODE((x), (y), (z)))

#endif // POGOSIM_VERSION_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
