
namespace slimt {

// NOLINTBEGIN
enum class VExt {
  w0,  //
  w1,  //
  w4,  //
  w8,  //
};

// NOLINTEND
template <enum VExt>
struct VDatum;

template <enum VExt>
struct Ops;

}  // namespace slimt

#ifdef __AVX__
#include "slimt/simd/avx2.h"

namespace slimt {
using F32x8 = VDatum<VExt::w8>;
}

#endif

#ifdef __SSE__
#include "slimt/simd/sse.h"

namespace slimt {
using F32x4 = VDatum<VExt::w4>;
}
#endif
