#pragma once
#include <iostream>

#include "slimt/TensorOps.hh"
#include "slimt/Utils.hh"
#include "slimt/slimt.hh"

#define CHECK_EQUAL(lhs, rhs, fn)       \
  do {                                  \
    bool pass = lhs == rhs;             \
    if (pass) {                         \
      std::cout << "[PASS]";            \
    } else {                            \
      std::cout << "[FAIL]";            \
    }                                   \
    std::cout << " " << fn << "\n";     \
    if (!pass) {                        \
      if (std::getenv("SLIMT_DEBUG")) { \
        diagnose(lhs, rhs);             \
      }                                 \
    }                                   \
  } while (0)

inline std::string blob_path(const std::string &bin) {
  const char *blob_path = std::getenv("SLIMT_BLOB_PATH");
  if (not blob_path) {
    std::cerr << "SLIMT_BLOB_PATH not define in environment.";
    std::exit(EXIT_FAILURE);
  }
  return std::string(blob_path) + '/' + bin;
}

inline void diagnose(const slimt::Tensor &lhs, const slimt::Tensor &rhs) {
  const auto *l = lhs.data<float>();
  const auto *r = rhs.data<float>();
  constexpr float kEps = 1e-9;
  size_t size = lhs.size();
  for (size_t i = 0; i < size; i++) {
    if (std::abs(l[i] - r[i]) > kEps) {
      fprintf(stdout, "values differ at %zu: %.9g %.9g, diff = %.9f\n", i, l[i],
              r[i], std::abs(l[i] - r[i]));
    }
  }
}
