#pragma once
#include <iostream>

#include "slimt/TensorOps.hh"
#include "slimt/Utils.hh"
#include "slimt/slimt.hh"

#define CHECK_EQUAL(lhs, rhs, fn)   \
  do {                              \
    if (lhs == rhs) {               \
      std::cout << "[PASS]";        \
    } else {                        \
      std::cout << "[FAIL]";        \
    }                               \
    std::cout << " " << fn << "\n"; \
  } while (0)

inline std::string blob_path(const std::string &bin) {
  const char *blob_path = std::getenv("SLIMT_BLOB_PATH");
  if (not blob_path) {
    std::cerr << "SLIMT_BLOB_PATH not define in environment.";
    std::exit(EXIT_FAILURE);
  }
  return std::string(blob_path) + '/' + bin;
}
