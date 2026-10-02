# Static link interfaces are not read by external Makefiles. Merge Context's
# archive members (or MSVC DLL import members) into the installed Pogosim archive.
foreach(required POGOSIM_ARCHIVE CONTEXT_ARCHIVE ARCHIVER ARCHIVE_STYLE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing ${required} for Boost.Context bundling")
    endif()
endforeach()
if(NOT EXISTS "${POGOSIM_ARCHIVE}" OR NOT EXISTS "${CONTEXT_ARCHIVE}")
    message(FATAL_ERROR "Pogosim and Boost.Context archives must exist before bundling")
endif()

# Build beside the original and replace it only after a successful merge. This
# also keeps failure from leaving a partly written installed/build library.
set(merged "${POGOSIM_ARCHIVE}.context-merged")
if(ARCHIVE_STYLE STREQUAL "MSVC")
    execute_process(COMMAND "${ARCHIVER}" /NOLOGO "/OUT:${merged}"
        "${POGOSIM_ARCHIVE}" "${CONTEXT_ARCHIVE}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
elseif(ARCHIVE_STYLE STREQUAL "APPLE")
    execute_process(COMMAND "${ARCHIVER}" -static -o "${merged}"
        "${POGOSIM_ARCHIVE}" "${CONTEXT_ARCHIVE}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
elseif(ARCHIVE_STYLE STREQUAL "MRI")
    # ADDLIB copies members; adding the archive as a single nested member would
    # not let a normal static linker find the Context symbols.
    if(NOT CONTEXT_ARCHIVE MATCHES "\\.a$")
        message(FATAL_ERROR "A static Boost.Context .a archive is required for bundling")
    endif()
    # GNU ar's MRI parser cannot quote paths containing spaces. Use simple
    # relative names in an owned scratch directory; WORKING_DIRECTORY itself
    # can contain spaces (common in Windows and external install prefixes).
    string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
    set(work "${POGOSIM_ARCHIVE}.context-${suffix}")
    while(EXISTS "${work}")
        string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
        set(work "${POGOSIM_ARCHIVE}.context-${suffix}")
    endwhile()
    file(MAKE_DIRECTORY "${work}")
    configure_file("${POGOSIM_ARCHIVE}" "${work}/pogosim.a" COPYONLY)
    configure_file("${CONTEXT_ARCHIVE}" "${work}/context.a" COPYONLY)
    file(WRITE "${work}/merge.mri"
        "CREATE merged.a\nADDLIB pogosim.a\nADDLIB context.a\nSAVE\nEND\n")
    execute_process(COMMAND "${ARCHIVER}" -M INPUT_FILE "${work}/merge.mri"
        WORKING_DIRECTORY "${work}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if("${status}" STREQUAL "0" AND EXISTS "${work}/merged.a")
        file(RENAME "${work}/merged.a" "${merged}")
    endif()
    file(REMOVE_RECURSE "${work}")
else()
    message(FATAL_ERROR "Unsupported Boost.Context archive style: ${ARCHIVE_STYLE}")
endif()
if(NOT "${status}" STREQUAL "0" OR NOT EXISTS "${merged}")
    message(FATAL_ERROR "Boost.Context bundling failed (${status}):\n${output}\n${error}")
endif()
file(RENAME "${merged}" "${POGOSIM_ARCHIVE}")
