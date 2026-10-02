# Check the exact exit status and diagnostic; WILL_FAIL alone also accepts
# unexpected crashes and missing runtime DLLs on native Windows.
execute_process(COMMAND "${PROGRAM}" -c "${CONFIG}" -g -nr
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT "${status}" STREQUAL "${EXPECTED_STATUS}")
    message(FATAL_ERROR "Expected exit ${EXPECTED_STATUS}, got ${status}\n${output}\n${error}")
endif()
if(DEFINED EXPECTED_MESSAGE AND NOT "${output}\n${error}" MATCHES "${EXPECTED_MESSAGE}")
    message(FATAL_ERROR "Missing diagnostic '${EXPECTED_MESSAGE}'\n${output}\n${error}")
endif()
