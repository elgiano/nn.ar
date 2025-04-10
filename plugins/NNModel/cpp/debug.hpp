#pragma once

namespace NN {
  extern bool gDebug;
}

#define Debug(...) if (gDebug) Print(__VA_ARGS__)

#include <iostream>
#include <cstdio>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define dup _dup
#define dup2 _dup2
#define STDERR_FILENO 2
#define STDOUT_FILENO 1
#define close _close
#else
#include <unistd.h>
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
