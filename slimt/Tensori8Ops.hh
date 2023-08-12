#include <cstddef>

#include "slimt/Tensor.hh"

#ifdef __x86_64__
namespace slimt {
Tensor intgemm_affine(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                      float b_quant, const std::string& name = "");
Tensor intgemm_affine_with_select(Tensor& x, Tensor& W, Tensor& b,
                                  float a_quant, float b_quant,
                                  const std::vector<uint32_t>& indices,
                                  const std::string& name = "");
Tensor intgemm_dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
                   const std::string& name = "");
}  // namespace slimt
#endif
