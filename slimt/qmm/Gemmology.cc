#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "gemmology/gemmology.h"
#pragma GCC diagnostic pop

#if defined(USE_AVX512)
#define GEMMOLOGY_SUPPORTED_ARCHS \
  xsimd::arch_list<xsimd::avx512bw, xsimd::avx2, xsimd::ssse3, xsimd::sse2>
#elif defined(USE_AVX2)
#define GEMMOLOGY_SUPPORTED_ARCHS \
  xsimd::arch_list<xsimd::avx2, xsimd::ssse3, xsimd::sse2>
#elif defined(USE_SSSE3)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::ssse3, xsimd::sse2>
#elif defined(USE_SSE2)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::sse2>
#elif defined(USE_NEON) and defined(XSIMD_WITH_NEON64)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::neon64>
#else
#error no supported architecture
#endif

namespace gemmology {

#ifdef USE_AVX512
template struct Engine<xsimd::avx512bw>;
template void Engine<xsimd::avx512bw>::SelectColumnsB(const int8_t*, int8_t*,
                                                      size_t, const uint32_t*,
                                                      const uint32_t*);
template void Engine<xsimd::avx512bw>::Shift::Multiply(
    const uint8_t*, const int8_t*, size_t, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
template void Engine<xsimd::avx512bw>::Shift::PrepareBias(
    const int8_t*, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
#endif  // USE_AVX512

#ifdef USE_AVX2
template struct Engine<xsimd::avx2>;
template void Engine<xsimd::avx2>::SelectColumnsB(const int8_t*, int8_t*,
                                                  size_t, const uint32_t*,
                                                  const uint32_t*);
template void Engine<xsimd::avx2>::Shift::Multiply(
    const uint8_t*, const int8_t*, size_t, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
template void Engine<xsimd::avx2>::Shift::PrepareBias(
    const int8_t*, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
#endif  // USE_AVX2

#ifdef USE_SSSE3
template struct Engine<xsimd::ssse3>;
template void Engine<xsimd::ssse3>::SelectColumnsB(const int8_t*, int8_t*,
                                                   size_t, const uint32_t*,
                                                   const uint32_t*);
template void Engine<xsimd::ssse3>::Shift::Multiply(
    const uint8_t*, const int8_t*, size_t, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
template void Engine<xsimd::ssse3>::Shift::PrepareBias(
    const int8_t*, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
#endif  // USE_SSSE3

#ifdef USE_SSE2
template struct Engine<xsimd::sse2>;
template void Engine<xsimd::sse2>::SelectColumnsB(const int8_t*, int8_t*,
                                                  size_t, const uint32_t*,
                                                  const uint32_t*);

template void Engine<xsimd::sse2>::Shift::Multiply(
    const uint8_t*, const int8_t*, size_t, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
template void Engine<xsimd::sse2>::Shift::PrepareBias(
    const int8_t*, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
#endif  // USE_SSE2

#ifdef USE_NEON
template struct Engine<xsimd::neon64>;
template void Engine<xsimd::neon64>::SelectColumnsB(const int8_t*, int8_t*,
                                                    size_t, const uint32_t*,
                                                    const uint32_t*);
template void Engine<xsimd::neon64>::Shift::Multiply(
    const uint8_t*, const int8_t*, size_t, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
template void Engine<xsimd::neon64>::Shift::PrepareBias(
    const int8_t*, size_t, size_t,
    gemmology::callbacks::UnquantizeAndAddBiasAndWrite);
#endif  // USE_NEON
}  // namespace gemmology
