#include <cstdint>
#include <cstdlib>

namespace slimt {

class Aligned {
 public:
  Aligned() = default;
  Aligned(size_t alignment, size_t size);

  ~Aligned();

  void* data() const;
  size_t size() const;

  char* begin() const;
  char* end() const;

  Aligned(const Aligned&) = delete;
  Aligned& operator=(const Aligned&) = delete;

  Aligned(Aligned&& from) noexcept;
  Aligned& operator=(Aligned&& from) noexcept;

 private:
  static void* allocate(size_t alignment, size_t size);
  void release();
  void consume(Aligned& from);

  void* data_ = nullptr;
  size_t size_ = 0;
};
}  // namespace slimt
