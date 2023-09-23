#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "slimt/Macros.hh"
#include "slimt/Types.hh"

namespace slimt {

template <class Key, class Value, class Hash = std::hash<Key>,
          class Equals = std::equal_to<Key>>
class AtomicCache {
 public:
  struct Stats {
    size_t hits{0};
    size_t misses{0};
  };

  explicit AtomicCache(size_t size, size_t buckets)
      : records_(size), mutexBuckets_(buckets) {}

  std::pair<bool, Value> find(const Key &key) const {
    Value value;
    bool found = atomicLoad(key, value);
    return std::make_pair(found, value);
  }

  void store(const Key &key, Value value) { atomicStore(key, value); }

  Stats stats() const {
#ifdef ENABLE_CACHE_STATS
    return Stats{hits_.load(), misses_.load()};
#else
    SLIMT_ABORT(
        "Cache statistics requested without enabling in builds. Please use "
        "-DENABLE_CACHE_STATS with cmake.");
    return Stats{0, 0};
#endif
  }

 private:
  using Record = std::pair<Key, Value>;

  bool atomicLoad(const Key &key, Value &value) const {
    // No probing, direct map onto records_
    size_t index = hash_(key) % records_.size();
    size_t mutex_id = index % mutexBuckets_.size();

    std::lock_guard<std::mutex> lock(mutexBuckets_[mutex_id]);
    const Record &candidate = records_[index];
    if (equals_(key, candidate.first)) {
      value = candidate.second;
#ifdef ENABLE_CACHE_STATS
      ++hits_;
#endif
      return true;
    }

#ifdef ENABLE_CACHE_STATS
    ++misses_;
#endif
    return false;
  }

  void atomicStore(const Key &key, Value value) {
    // No probing, direct map onto records_
    size_t index = hash_(key) % records_.size();
    size_t mutex_id = index % mutexBuckets_.size();

    std::lock_guard<std::mutex> lock(mutexBuckets_[mutex_id]);
    Record &candidate = records_[index];

    candidate.first = key;
    candidate.second = value;
  }

  std::vector<Record> records_;

  mutable std::vector<std::mutex> mutexBuckets_;

#ifdef ENABLE_CACHE_STATS
  mutable std::atomic<size_t> hits_{0};
  mutable std::atomic<size_t> misses_{0};
#endif

  Hash hash_;
  Equals equals_;
};

using TranslationCache = AtomicCache<size_t, History>;

}  // namespace slimt
