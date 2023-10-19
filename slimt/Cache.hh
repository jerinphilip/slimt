#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace slimt {

template <class Key, class Value, class Hash = std::hash<Key>,
          class Equals = std::equal_to<Key>>
class AtomicCache {
 public:
  explicit AtomicCache(size_t size, size_t buckets)
      : records_(size), locks_(buckets) {}

  std::pair<bool, Value> find(const Key &key) const {
    Value value;
    bool found = atomic_load(key, value);
    return std::make_pair(found, value);
  }

  void store(const Key &key, Value value) { atomic_store(key, value); }

 private:
  using Record = std::pair<Key, Value>;

  bool atomic_load(const Key &key, Value &value) const {
    // No probing, direct map onto records_
    size_t index = hash_(key) % records_.size();
    size_t lock_id = index % locks_.size();

    std::lock_guard<std::mutex> guard(locks_[lock_id]);
    const Record &candidate = records_[index];
    if (equals_(key, candidate.first)) {
      value = candidate.second;
      return true;
    }

    return false;
  }

  void atomic_store(const Key &key, Value value) {
    // No probing, direct map onto records_
    size_t index = hash_(key) % records_.size();
    size_t lock_id = index % locks_.size();

    std::lock_guard<std::mutex> guard(locks_[lock_id]);
    Record &candidate = records_[index];

    candidate.first = key;
    candidate.second = value;
  }

  std::vector<Record> records_;
  mutable std::vector<std::mutex> locks_;
  Hash hash_;
  Equals equals_;
};

}  // namespace slimt
