#pragma once

namespace NN {
  extern bool gDebug;
}

#define Debug(...) if (gDebug) Print(__VA_ARGS__)

#include <iostream>
#include <cstdio>
#if defined(_WIN32) && !defined(__MINGW32__)
#   include <io.h>
#   include <fcntl.h>
    static inline int dup(int fd) { return _dup(fd); }
    static inline int dup2(int fd1, int fd2) { return _dup2(fd1, fd2); }
    static inline int close(int fd) { return _close(fd); }
#   define STDERR_FILENO 2
#   define STDOUT_FILENO 1
#else
#   include <unistd.h>
#endif

// we use this to redirect libtorch errors to stdout
class StdErr2StdOut {
    int original_stderr;
    
public:
    StdErr2StdOut() {
        // Save original stderr and redirect to stdout
        original_stderr = dup(STDERR_FILENO);
        dup2(STDOUT_FILENO, STDERR_FILENO);
    }
    
    ~StdErr2StdOut() {
        // Flush before restoring
        std::cerr.flush();
        std::fflush(stderr);
        dup2(original_stderr, STDERR_FILENO);
        close(original_stderr);
    }
};
