#pragma once
#include <cstdio>
#include <iostream>

#define SLIMT_BREAK std::raise(SIGTRAP)

#define SLIMT_TRACE(x)                                         \
  do {                                                         \
    std::cerr << __FILE__ << ":" << __LINE__;                  \
    std::cerr << " " << __FUNCTION__ << " ";                   \
    std::cerr << #x << ": " << std::scientific << (x) << '\n'; \
  } while (0)

#define SLIMT_TRACE_BLOCK(x)                                   \
  do {                                                         \
    std::cerr << __FILE__ << ":" << __LINE__;                  \
    std::cerr << " " << __FUNCTION__ << " \n\n";               \
    std::cerr << #x << ": " << std::scientific << (x) << '\n'; \
    std::cerr << "\n\n";                                       \
  } while (0);

#define SLIMT_TRACE2(x, y) \
  SLIMT_TRACE(x);          \
  SLIMT_TRACE(y)

#define SLIMT_TRACE3(x, y, z) \
  SLIMT_TRACE2(x, y);         \
  SLIMT_TRACE(z);

#define SLIMT_ABORT_IF(condition, error) \
  do {                                   \
    if (condition) {                     \
      std::cerr << (error) << '\n';      \
      std::abort();                      \
    }                                    \
  } while (0)

#define SLIMT_ABORT(message) \
  do {                       \
    std::cerr << (message);  \
    std::abort();            \
  } while (0)

#ifdef SLIMT_ENABLE_LOG
#define LOG(level, ...)              \
  do {                               \
    fprintf(stderr, "[%s]", #level); \
    fprintf(stderr, __VA_ARGS__);    \
    fprintf(stderr, "\n");           \
  } while (0)
#else  // SLIMT_ENABLE_LOGS
#define LOG(...) (void)0
#endif  // SLIMT_ENABLE_LOGS
