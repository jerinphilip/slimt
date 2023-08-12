
namespace slimt {

template <class Type>
struct Ops;

}

#ifdef __AVX__
#include "slimt/simd/avx2.h"
#endif

#ifdef __SSE__
#include "slimt/simd/sse.h"
#endif
