if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_BRANCH_OR_TAG STREQUAL "")
        set(GIT_TAG "no-tag")
    endif()
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    SET(NNAR_VERSION "${GIT_TAG}::${GIT_COMMIT_HASH}")
else()
    SET(NNAR_VERSION not_a_git_checkout)
endif()
