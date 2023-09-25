#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#include "slimt/slimt.hh"

namespace slimt {

#define SLIMT_CHECK(condition)                                                \
  do {                                                                        \
    if (!(condition)) {                                                       \
      fprintf(stderr, "%s:%d %s failed\n", __FILE__, __LINE__, (#condition)); \
      throw std::runtime_error("Failed test");                                \
    }                                                                         \
    fprintf(stderr, "%s:%d %s success\n", __FILE__, __LINE__, (#condition));  \
  } while (0)

static const std::string kBlobPath = checked_fpath();

namespace {

std::string prefix(const std::string &fname) { return kBlobPath + "/" + fname; }

template <typename Scalar, class... Args>
Tensor tf(const std::string &path, Args &&...args) {
  return tensor_from_file<Scalar>(prefix(path), std::forward<Args>(args)...);
}

template <typename Scalar, typename Quant, class... Args>
std::tuple<Tensor, float> qtf(const std::string &path, Args &&...args) {
  return quantized_tensor_from_file<Scalar, Quant>(prefix(path),
                                                   std::forward<Args>(args)...);
}

}  // namespace
   //
   //

void load() {
  std::string fname = "RowsNodeOp-float32_8x256-rhs1-uint32_8_data_0.bin";
  Tensor x = tf<int>((fname), Shape({8}), "rhs1");
  auto *data = x.data<int>();
  // std::cout << x << "\n";

  float begin = *data;
  float rbegin = *(data + (x.size() - 1));
  SLIMT_CHECK(begin == 39);
  SLIMT_CHECK(rbegin == 0);
}

struct OpArgs {
  std::string lhs;
  std::vector<std::string> rhs;
};

void ScalarMultNodeOp() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/graph/node_operators_unary.h"
  // line: 100
  // fn: "marian::ScalarMultNodeOp::forwardOps()::<lambda()>"
  // op: { Element(_1 = scalar_ * _2, val_, child(0)->val()) }
  // before: var_5 float32 [4x2x256]
  // after: var_5 float32 [4x2x256] ScalarMultNodeOp-float32_4x2x256-lhs.bin
  // operands:
  //    - var_3 float32 [4x2x256] ScalarMultNodeOp-float32_4x2x256-rhs0-float32_4x2x256.bin
  // clang-format on

  OpArgs args{
      .lhs = "ScalarMultNodeOp-float32_4x2x256-lhs.bin",                    //
      .rhs = {"ScalarMultNodeOp-float32_4x2x256-rhs0-float32_4x2x256.bin"}  //
  };

  Shape shape({4, 2, 256});
  Tensor lhs = tf<float>((args.lhs), shape, "lhs");
  Tensor rhs = tf<float>((args.rhs[0]), Shape({4, 2, 256}), "rhs");

  Tensor lhs_computed(lhs.type(), lhs.shape(), "lhs_computed");

  float embedding_dim_sqrt = std::sqrt(256.0F);
  mul_scalar(rhs.data<float>(), embedding_dim_sqrt, rhs.size(),
             lhs_computed.data<float>());

  // std::cout << rhs << "\n";
  // std::cout << lhs << "\n" << lhs_computed << "\n";

  SLIMT_CHECK(lhs_computed == lhs);
}

void RowsNodeOp() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/graph/node_operators_binary.h"
  // line: 672
  // fn: "marian::RowsNodeOp::forwardOps()::<lambda()>"
  // op: { CopyRows(val_, child(0)->val(), child(1)->val()) }
  // before: var_2 float32 [8x256]
  // after: var_2 float32 [8x256] RowsNodeOp-float32_8x256-lhs.bin
  // operands:
  //   - var_0 float32 [32000x256] RowsNodeOp-float32_8x256-rhs0-float32_32000x256.bin
  //   - var_1 uint32 [8] RowsNodeOp-float32_8x256-rhs1-uint32_8.bin
  // clang-format on

  OpArgs args{
      .lhs = "RowsNodeOp-float32_8x256-lhs.bin",  //
      .rhs =
          {
              "RowsNodeOp-float32_8x256-rhs0-float32_32000x256_Wemb.bin",  //
              "RowsNodeOp-float32_8x256-rhs1-uint32_8_data_0.bin"          //
          }                                                                //
  };

  // Shape projected to 1 x 8 to match.
  Tensor lhs = tf<float>((args.lhs), Shape({1, 8, 256}), "lhs");
  // std::cout << "\n" << lhs << std::endl;

  Tensor rhs0 = tf<float>((args.rhs[0]), Shape({32000, 256}), "rhs0");

  // std::cout << rhs0 << std::endl;

  Tensor rhs1 = tf<int>((args.rhs[1]), Shape({1, 8}), "rhs1");

  // std::cout << rhs1 << std::endl;

  Tensor lhs_computed = index_select(rhs0, rhs1);
  SLIMT_CHECK(lhs_computed == lhs);
}

void DotBatchedNodeOp() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/graph/node_operators_binary.h"
  // line: 424
  // fn: "marian::DotBatchedNodeOp::forwardOps()::<lambda()>"
  // op: { ProdBatched(val_, graph()->allocator(), child(0)->val(), child(1)->val(), transA_, transB_, 0.f, scalar_) }
  // before: var_44 float32 [2x8x4x4]
  // after: var_44 float32 [2x8x4x4] DotBatchedNodeOp-float32_2x8x4x4-lhs.bin
  // operands: 
  //   - var_25 float32 [2x8x4x32] DotBatchedNodeOp-float32_2x8x4x4-rhs0-float32_2x8x4x32.bin
  //   - var_34 float32 [2x8x4x32] DotBatchedNodeOp-float32_2x8x4x4-rhs1-float32_2x8x4x32.bin
  // clang-format on

  OpArgs args{
      .lhs = "DotBatchedNodeOp-float32_2x8x4x4-lhs.bin",  //
      .rhs =
          {
              "DotBatchedNodeOp-float32_2x8x4x4-rhs0-float32_2x8x4x32.bin",  //
              "DotBatchedNodeOp-float32_2x8x4x4-rhs1-float32_2x8x4x32.bin"   //
          }                                                                  //
  };

  // std::cout << "\n";

  size_t batch_size = 2;
  size_t sequence_length = 4;
  size_t num_heads = 8;
  size_t dim_head = 32;

  size_t k = 2;
  size_t h = num_heads / k;
  Shape lhs_shape({k, batch_size * sequence_length, h, h});

  Shape rhs_shape({k, batch_size * sequence_length, h, dim_head});
  Tensor lhs = tf<float>((args.lhs), lhs_shape, "lhs");
  // std::cout << lhs << std::endl;

  Tensor rhs0 = tf<float>((args.rhs[0]), rhs_shape, "rhs0");
  // std::cout << rhs0 << std::endl;

  Tensor rhs1 = tf<float>((args.rhs[1]), rhs_shape, "rhs1");
  // std::cout << rhs1 << std::endl;

  // clang-format off
  // op: { ProdBatched(val_, graph()->allocator(), child(0)->val(), child(1)->val(), transA_, transB_, 0.f, scalar_) }
  //                                                                                 false    true          0.176776692
  // clang-format on

  size_t bsz = batch_size * sequence_length * k;
  Tensor lhs_computed(lhs.type(), lhs.shape(), "lhs_computed");
  batch_matrix_multiply(                       //
      rhs0.data<float>(), rhs1.data<float>(),  //
      bsz, h, dim_head, h, dim_head,           //
      /*trans_a=*/false, /*trans_b=*/true,     //
      /*alpha =*/0.176776692,                  //
      lhs_computed.data<float>());

  // std::cout << lhs << std::endl;
  // std::cout << lhs_computed << std::endl;
  SLIMT_CHECK(lhs_computed == lhs);
}

void TransposeNodeOp() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/graph/node_operators_unary.h"
  // line: 747
  // fn: "marian::TransposeNodeOp::forwardOps()::<lambda()>"
  // op: { TransposeND(val_, child(0)->val(), axes_) }
  // before: var_10 float32 [1x2x4x256]
  // after: var_10 float32 [1x2x4x256] TransposeNodeOp-float32_1x2x4x256-lhs.bin
  // operands: 
  //   - var_8 float32 [1x4x2x256] TransposeNodeOp-float32_1x2x4x256-rhs0-float32_1x4x2x256.bin
  // clang-format on

  OpArgs args{
      .lhs = "TransposeNodeOp-float32_1x2x4x256-lhs.bin",
      .rhs = {"TransposeNodeOp-float32_1x2x4x256-rhs0-float32_1x4x2x256.bin"}};

  Shape lhs_shape({1, 2, 4, 256});
  Tensor lhs = tf<float>((args.lhs), lhs_shape, "lhs");

  Shape rhs_shape = lhs_shape.transpose(-1, -2);
  Tensor rhs = tf<float>((args.rhs[0]), rhs_shape, "rhs");

  Tensor lhs_expected(lhs.type(), lhs.shape(), "lhs_expected");
  transpose_3120(rhs.data<float>(), 1, 4, 2, 256, lhs_expected.data<float>());

  SLIMT_TRACE(lhs);
  SLIMT_TRACE(lhs_expected);
  SLIMT_CHECK(lhs == lhs_expected);
}

void LayerNormalizationOp() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/graph/node_operators_binary.h"
  // line: 1210
  // fn: "marian::LayerNormalizationOp::forwardOps()::<lambda()>"
  // op: { LayerNormalization(val_, child(0)->val(), child(1)->val(), (children_.size() == 3) ? child(2)->val() : nullptr, eps_) }
  // before: var_60 float32 [1x2x4x256]
  // after: var_60 float32 [1x2x4x256] LayerNormalizationOp-float32_1x2x4x256-lhs.bin
  // operands: 
  //   - var_57 float32 [1x2x4x256] LayerNormalizationOp-float32_1x2x4x256-rhs0-float32_1x2x4x256.bin
  //   - var_58 float32 [1x256] F0::encoder_l1_self_Wo_ln_scale LayerNormalizationOp-float32_1x2x4x256-rhs1-float32_1x256_encoder_l1_self_Wo_ln_scale.bin
  //   - var_59 float32 [1x256] F0::encoder_l1_self_Wo_ln_bias LayerNormalizationOp-float32_1x2x4x256-rhs2-float32_1x256_encoder_l1_self_Wo_ln_bias.bin
  // clang-format on
  OpArgs args{
      .lhs = "LayerNormalizationOp-float32_1x2x4x256-lhs.bin",
      // clang-format off
      .rhs = {
          "LayerNormalizationOp-float32_1x2x4x256-rhs0-float32_1x2x4x256.bin",  
          "LayerNormalizationOp-float32_1x2x4x256-rhs1-float32_1x256_encoder_l1_self_Wo_ln_scale.bin", 
          "LayerNormalizationOp-float32_1x2x4x256-rhs2-float32_1x256_encoder_l1_self_Wo_ln_bias.bin" 
      }
      // clang-format on
  };

  Shape lhs_shape({1, 2, 4, 256});
  Tensor lhs = tf<float>((args.lhs), lhs_shape, "lhs");

  Tensor rhs0 = tf<float>((args.rhs[0]), lhs_shape, "rhs0");

  Shape ln_shape({1, 256});
  Tensor rhs1 = tf<float>((args.rhs[1]), ln_shape, "rhs1");
  Tensor rhs2 = tf<float>((args.rhs[2]), ln_shape, "rhs2");

  Tensor lhs_expected(lhs.type(), lhs.shape(), "lhs_expected");
  constexpr float kEps = 1e-9;
  size_t rows = 1 * 2 * 4;
  size_t cols = 256;

  size_t gamma_stride = 1;
  size_t beta_stride = 1;
  bool has_beta = true;
  layer_norm(rhs0.data<float>(), rhs1.data<float>(), rhs2.data<float>(), kEps,
             rows, cols, gamma_stride, beta_stride, has_beta,
             lhs_expected.data<float>());

  SLIMT_TRACE(lhs);
  SLIMT_TRACE(lhs_expected);
  SLIMT_CHECK(lhs == lhs_expected);
}
}  // namespace slimt

#ifdef HAS_INTGEMM
#include "3rd-party/intgemm/intgemm/intgemm.h"
namespace slimt {

void AffineIntgemm() {
  // clang-format off
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/tensors/cpu/integer_common.h"
  // line: 55
  // fn: "marian::cpu::integer::fetchAlphaFromModelNodeOp::forwardOps()::<lambda()>"
  // op: { fetchAlpha() }
  // before: var_19 float32 [1] F0::encoder_l1_self_Wq_QuantMultA
  // after: var_19 float32 [1] F0::encoder_l1_self_Wq_QuantMultA cpu-float32_1_encoder_l1_self_Wq_QuantMultA-lhs.bin
  // operands: 
  //   - var_17 intgemm8 [256x256] F0::encoder_l1_self_Wq cpu-float32_1_encoder_l1_self_Wq_QuantMultA-rhs0-intgemm8_256x256_encoder_l1_self_Wq.bin
  // 
  // 
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/tensors/cpu/intgemm_interface.h"
  // line: 60
  // fn: "marian::cpu::integer::PrepareANodeOp<marian::Type::int8>::forwardOps()::<lambda()>"
  // op: { PrepareA() }
  // before: var_20 int8 [1x2x4x256] none_shifted
  // after: var_20 int8 [1x2x4x256] none_shifted cpu-int8_1x2x4x256_none_shifted-lhs.bin
  // operands: 
  //   - var_10 float32 [1x2x4x256] cpu-int8_1x2x4x256_none_shifted-rhs0-float32_1x2x4x256.bin
  //   - var_19 float32 [1] F0::encoder_l1_self_Wq_QuantMultA cpu-int8_1x2x4x256_none_shifted-rhs1-float32_1_encoder_l1_self_Wq_QuantMultA.bin
  // 
  // 
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/tensors/cpu/intgemm_interface.h"
  // line: 285
  // fn: "marian::cpu::integer::QuantMultNodeOp<marian::Type::int8>::forwardOps()::<lambda()>"
  // op: { QuantMult() }
  // before: var_21 float32 [1] F0::encoder_l1_self_Wq_QuantMultB
  // after: var_21 float32 [1] F0::encoder_l1_self_Wq_QuantMultB cpu-float32_1_encoder_l1_self_Wq_QuantMultB-lhs.bin
  // operands: 
  //   - var_17 intgemm8 [256x256] F0::encoder_l1_self_Wq cpu-float32_1_encoder_l1_self_Wq_QuantMultB-rhs0-intgemm8_256x256_encoder_l1_self_Wq.bin
  // 
  // 
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/tensors/cpu/intgemm_interface.h"
  // line: 359
  // fn: "marian::cpu::integer::PrepareBiasForBNodeOp::forwardOps()::<lambda()>"
  // op: { PrepareBias() }
  // before: var_22 float32 [1x256] F0::encoder_l1_self_bq_Prepared
  // after: var_22 float32 [1x256] F0::encoder_l1_self_bq_Prepared cpu-float32_1x256_encoder_l1_self_bq_Prepared-lhs.bin
  // operands: 
  //   - var_18 float32 [1x256] F0::encoder_l1_self_bq cpu-float32_1x256_encoder_l1_self_bq_Prepared-rhs0-float32_1x256_encoder_l1_self_bq.bin
  //   - var_17 intgemm8 [256x256] F0::encoder_l1_self_Wq cpu-float32_1x256_encoder_l1_self_bq_Prepared-rhs1-intgemm8_256x256_encoder_l1_self_Wq.bin
  //   - var_19 float32 [1] F0::encoder_l1_self_Wq_QuantMultA cpu-float32_1x256_encoder_l1_self_bq_Prepared-rhs2-float32_1_encoder_l1_self_Wq_QuantMultA.bin
  //   - var_21 float32 [1] F0::encoder_l1_self_Wq_QuantMultB cpu-float32_1x256_encoder_l1_self_bq_Prepared-rhs3-float32_1_encoder_l1_self_Wq_QuantMultB.bin
  // 
  // 
  // quantmult A, B, scalar 1.007505e+01 1.823447e+02 1.000000e+00
  // file: "/home/jerin/code/bergamot-translator/3rd_party/marian-dev/src/tensors/cpu/intgemm_interface.h"
  // line: 540
  // fn: "marian::cpu::integer::AffineNodeOp<marian::Type::int8>::forwardOps()::<lambda()>"
  // op: { AffineOp() }
  // before: var_23 float32 [1x2x4x256]
  // after: var_23 float32 [1x2x4x256] cpu-float32_1x2x4x256-lhs.bin
  // operands: 
  //   - var_20 int8 [1x2x4x256] none_shifted cpu-float32_1x2x4x256-rhs0-int8_1x2x4x256_none_shifted.bin
  //   - var_17 intgemm8 [256x256] F0::encoder_l1_self_Wq cpu-float32_1x2x4x256-rhs1-intgemm8_256x256_encoder_l1_self_Wq.bin
  //   - var_22 float32 [1x256] F0::encoder_l1_self_bq_Prepared cpu-float32_1x2x4x256-rhs2-float32_1x256_encoder_l1_self_bq_Prepared.bin
  // clang-format on

  // Input to the intgemm involved pipeline. Usually these are float
  // activations.
  //
  // Weight and weights quantization multiplier that corresponds to the x above.
  // Here it happens to be the encoder1's Q matrix.
  // Bias associated with the Q transform.

  // Intgemm code test, step-by-step. B is already prepared offline.
  //
  //   0. (offline) PrepareB
  //   1. PrepareA
  //   2. PrepareBias
  //   3. Multiply.

  // Aliases in intgemm terminology
  //  C = AB + bias .
  //      A  i8 [ A_rows x width ]
  //      B  i8 [ width x B_cols ]
  //   bias f32 [ 1 x B_cols     ]
  //

  // We define the following two structs to hold objects to give a convenient
  // syntax to describe for cases ahead.
  //
  // There are 3 variable sets:
  //
  // 1. Expected (raw):
  //      The unprepared values, that are fed in. In our case, it's offline
  //      prepared weights (B), f32 activations and f32 biases for online
  //      prepration.
  //
  // 2. Expected (prepared)
  //      During the course, we get prepared variations that are intermediate
  //      variables. These are also saved and requires ground truth to check
  //      expected.
  //
  // 3. Computed (prepared)
  //      The values we compute along the process.

  struct Affine {
    Tensor A;
    Tensor B;
    Tensor bias;
  };

  // Holds a, b scalar (hyper) parameters used to multiply or divide for
  // quantization.
  struct Quant {
    float a;
    float b;
  };

  struct ProblemSet {
    Affine var;
    Affine prepared_expected;
    Quant quant;
    Tensor y_expected;
  };

  auto problem_256x256 = []() {
    // clang-format off
    auto A            = tf<float>("cpu-int8_1x2x4x256_none_shifted-rhs0-float32_1x2x4x256.bin",         Shape({1*2*4, 256}), "A"); // NOLINT
    auto [B, qB]      = qtf<int8_t, float>("var_17-ParamNode-intgemm8_256x256_encoder_l1_self_Wq-lhs.bin", Shape({256, 256}), "B");
    auto bias         = tf<float>("var_18-ParamNode-float32_1x256_encoder_l1_self_bq-lhs.bin", Shape({1, 256}), "bias");
    auto qa           = tf<float>("var_19-cpu-float32_1_encoder_l1_self_Wq_QuantMultA-lhs.bin", Shape({1}), "quant.a"); // DONE
    auto qb           = tf<float>("cpu-float32_1x1536_encoder_l1_ffn_b1_Prepared-rhs3-float32_1_encoder_l1_ffn_W1_QuantMultB.bin", Shape({1}), "quant.b"); // DONE
    auto y_expected   = tf<float>("cpu-float32_1x2x4x256-lhs.bin", Shape({1*2*4, 256}), "y_expected");

    Affine prepared_expected {
      .A = tf<int8_t>("var_20-cpu-int8_1x2x4x256_none_shifted-lhs.bin", Shape({1*2*4, 256}), "prepared_expected_A"),
      .B = tf<int8_t>("var_17-ParamNode-intgemm8_256x256_encoder_l1_self_Wq-lhs.bin", Shape({256, 256}), "prepared_expected_B"),
      .bias= tf<float>("var_22-cpu-float32_1x256_encoder_l1_self_bq_Prepared-lhs.bin", Shape({1, 256}), "prepared_expected_bias")
    };
    // clang-format on

    ProblemSet pset{
        .var =
            Affine{
                .A = std::move(A),                          //
                .B = std::move(B),                          //
                .bias = std::move(bias)                     //
            },                                              //
        .prepared_expected = std::move(prepared_expected),  //
        .quant =
            Quant{
                .a = qa.item<float>(),       //
                .b = qB                      //
            },                               //
        .y_expected = std::move(y_expected)  //
    };

    // auto qb_loaded = qb.item<float>();
    // float diff = qb_loaded - qB;
    // SLIMT_TRACE3(qB, qb_loaded, diff);
    // SLIMT_CHECK(std::abs(diff) < 1e-7);
    // SLIMT_TRACE2(quant.a, quant.b);

    return pset;
  };

  auto problem_256x1536 = []() {
    // clang-format off
    auto A            = tf<float>("var_64-cpu-int8_1x2x4x256_none_shifted-rhs0-float32_1x2x4x256.bin",         Shape({2, 4, 256}), "A"); // NOLINT
    auto [B, qB]      = qtf<int8_t, float>("var_61-ParamNode-intgemm8_256x1536_encoder_l1_ffn_W1-lhs.bin", Shape({256, 1536}), "B");
    auto bias         = tf<float>("var_62-ParamNode-float32_1x1536_encoder_l1_ffn_b1-lhs.bin", Shape({1, 1536}), "bias");
    auto qa           = tf<float>("var_63-cpu-float32_1_encoder_l1_ffn_W1_QuantMultA-lhs.bin", Shape({1}), "quant.a");
    auto qb           = tf<float>("var_65-cpu-float32_1_encoder_l1_ffn_W1_QuantMultB-lhs.bin", Shape({1}), "quant.b");
    auto y_expected   = tf<float>("var_67-cpu-float32_1x2x4x1536-lhs.bin", Shape({2, 4, 1536}), "y_expected");

    Affine prepared_expected {
      .A = tf<int8_t>("var_64-cpu-int8_1x2x4x256_none_shifted-lhs.bin", Shape({2, 4, 256}), "prepared_expected_A"),
      .B = tf<int8_t>("var_66-cpu-float32_1x1536_encoder_l1_ffn_b1_Prepared-rhs1-intgemm8_256x1536_encoder_l1_ffn_W1.bin", Shape({256, 1536}), "prepared_expected_B"),
      .bias= tf<float>("var_66-cpu-float32_1x1536_encoder_l1_ffn_b1_Prepared-lhs.bin", Shape({1, 1536}), "prepared_expected_bias")
    };
    // clang-format on

    ProblemSet pset{
        .var =
            Affine{
                .A = std::move(A),                          //
                .B = std::move(B),                          //
                .bias = std::move(bias)                     //
            },                                              //
        .prepared_expected = std::move(prepared_expected),  //
        .quant =
            Quant{
                .a = qa.item<float>(),       //
                .b = qB                      //
            },                               //
        .y_expected = std::move(y_expected)  //
    };
    return pset;
  };

  auto problem_1536x256 = []() {
    // clang-format off
    auto A            = tf<float>("var_72-cpu-int8_1x2x4x1536_none_shifted-rhs0-float32_1x2x4x1536.bin",         Shape({2, 4, 1536}), "A"); // NOLINT
    auto [B, qB]      = qtf<int8_t, float>("var_69-ParamNode-intgemm8_1536x256_encoder_l1_ffn_W2-lhs.bin", Shape({1536, 256}), "B");
    auto bias         = tf<float>("var_70-ParamNode-float32_1x256_encoder_l1_ffn_b2-lhs.bin", Shape({1, 256}), "bias");
    auto qa           = tf<float>("var_71-cpu-float32_1_encoder_l1_ffn_W2_QuantMultA-lhs.bin", Shape({1}), "quant.a");
    auto qb           = tf<float>("var_73-cpu-float32_1_encoder_l1_ffn_W2_QuantMultB-lhs.bin", Shape({1}), "quant.b");
    auto y_expected   = tf<float>("var_75-cpu-float32_1x2x4x256-lhs.bin", Shape({2, 4, 256}), "y_expected");

    Affine prepared_expected {
      .A = tf<int8_t>("var_72-cpu-int8_1x2x4x1536_none_shifted-lhs.bin", Shape({2, 4, 1536}), "prepared_expected_A"),
      .B = tf<int8_t>("var_69-ParamNode-intgemm8_1536x256_encoder_l1_ffn_W2-lhs.bin", Shape({1536, 256}), "prepared_expected_B"),
      .bias= tf<float>("var_74-cpu-float32_1x256_encoder_l1_ffn_b2_Prepared-lhs.bin", Shape({1, 256}), "prepared_expected_bias")
    };
    // clang-format on

    ProblemSet pset{
        .var =
            Affine{
                .A = std::move(A),                          //
                .B = std::move(B),                          //
                .bias = std::move(bias)                     //
            },                                              //
        .prepared_expected = std::move(prepared_expected),  //
        .quant =
            Quant{
                .a = qa.item<float>(),       //
                .b = qB                      //
            },                               //
        .y_expected = std::move(y_expected)  //
    };
    return pset;
  };

  auto intgemm_from_params = [](ProblemSet &pset) {
    Affine &actual = pset.var;
    Affine &prepared_expected = pset.prepared_expected;
    Quant &quant = pset.quant;
    Tensor &y_expected = pset.y_expected;

    Affine prepared{
        .A = Tensor(Type::i8, actual.A.shape(), "prepared_A"),           //
        .B = Tensor(Type::i8, actual.B.shape(), "prepared_B"),           //
        .bias = Tensor(Type::f32, actual.bias.shape(), "prepared_bias")  //
    };

    size_t A_cols = actual.A.dim(-1);          // NOLINT
    size_t B_cols = actual.B.dim(-1);          // NOLINT
    size_t A_rows = actual.A.size() / A_cols;  // NOLINT
    size_t B_rows = actual.B.size() / B_cols;  // NOLINT

    // A is in row-major format.
    // B is in column-major, so consider it a transposed form.
    SLIMT_TRACE2(A_rows, A_cols);
    SLIMT_TRACE2(B_rows, B_cols);

    SLIMT_CHECK(A_cols == B_rows);
    size_t width = B_rows;
    SLIMT_TRACE(width);

    // Check widths are consistent, making matrix multiplication viable.
    // This ensures our saves and loads satisfy one property.

    // Now we proceed to piecewise intgemm operations.

    // 0. PrepareB: B is prepared, but let's check PrepareB.
    //
    // Turns out, I do not have access from inputs to the raw B. I already
    // only have prepared B.
    // TODO(jerinphilip): Come back later and fix.
    std::copy(actual.B.data<int8_t>(), actual.B.data<int8_t>() + B_cols * width,
              prepared.B.data<int8_t>());

    // Surprisingly, the following does not work. However a plain-copy does
    // work.
    // @jerinphilip has confirmed this is not a no-op by trying a copy before
    // (see above).
    //
    // const auto *b = B.data<int8_t>();
    // auto *prepared_b = prepared.B.data<int8_t>();
    // intgemm::Int8::PrepareBQuantizedTransposed(b, prepared_b, B_cols, width);

    // SLIMT_TRACE_BLOCK(prepared.B);
    // SLIMT_TRACE_BLOCK(prepared_expected.B);
    SLIMT_CHECK(prepared_expected.B == actual.B);
    SLIMT_CHECK(prepared.B == prepared_expected.B);

    // 1. PrepareA
    intgemm::Int8Shift::PrepareA(                           //
        actual.A.data<float>(), prepared.A.data<int8_t>(),  //
        quant.a,                                            //
        A_rows, width                                       //
    );

    // Check that the quantized activations are a match.
    // SLIMT_TRACE2(qx, A);
    SLIMT_CHECK(prepared.A == prepared_expected.A);

    // 2. PrepareBias
    Quant alpha{
        .a = 127.0F / quant.a,  //
        .b = 127.0F / quant.b,  //
    };

    float bias_unquant_multiplier = (-1.0F * (alpha.a * alpha.b)) / 127.0F;
    SLIMT_TRACE3(alpha.a, alpha.b, bias_unquant_multiplier);
    auto prepare_bias_callback =
        intgemm::callbacks::UnquantizeAndAddBiasAndWrite(        //
            bias_unquant_multiplier, actual.bias.data<float>(),  //
            prepared.bias.data<float>()                          //
        );

    SLIMT_TRACE2(width, B_cols);
    intgemm::Int8Shift::PrepareBias(  //
        prepared.B.data<int8_t>(),    //
        width, B_cols,                //
        prepare_bias_callback         //
    );

    SLIMT_TRACE_BLOCK(prepared.bias)
    SLIMT_TRACE_BLOCK(prepared_expected.bias);
    SLIMT_TRACE(mse(prepared.bias, prepared_expected.bias));
    SLIMT_CHECK(prepared.bias == prepared_expected.bias);

    // 3. Multiply
    Shape out_shape = actual.A.shape();
    out_shape.set_dim(-1, B_cols);

    Tensor y_piecewise(Type::f32, out_shape, "y_piecewise");

    float unquant_multiplier = 1.0F / (quant.a * quant.b);
    auto multiply_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
        unquant_multiplier, prepared.bias.data<float>(),
        y_piecewise.data<float>());

    intgemm::Int8Shift::Multiply(                              //
        prepared.A.data<int8_t>(), prepared.B.data<int8_t>(),  //
        A_rows, width, B_cols,                                 //
        multiply_callback                                      //
    );

    SLIMT_TRACE_BLOCK(y_piecewise);
    SLIMT_TRACE_BLOCK(y_expected);
    SLIMT_TRACE(mse(y_piecewise, y_expected));
    SLIMT_CHECK(y_expected == y_piecewise);

    // Compute from the intgemm_affine function, used in the library.
    // This ensures what we checked in there is consistent with what we expect.
    Tensor y_whole = qmm::affine(actual.A, actual.B, actual.bias, quant.a,
                                 quant.b, "y_whole");
    SLIMT_TRACE(y_whole.shape());
    SLIMT_TRACE(y_expected.shape());
    SLIMT_TRACE(mse(y_whole, y_expected));
    SLIMT_CHECK(y_expected == y_whole);
  };

  auto pset1 = problem_256x256();
  auto pset2 = problem_256x1536();
  auto pset3 = problem_1536x256();
  intgemm_from_params(pset1);
  intgemm_from_params(pset2);
  intgemm_from_params(pset3);

  // SLIMT_TRACE2(y_whole, y_expected);
}
}  // namespace slimt
#endif

namespace slimt {
template <class Field>
struct Record {
  Field model;
  Field vocabulary;
  Field shortlist;
};

void integration() {
  std::string home = std::getenv("HOME");
  std::string browsermt = ".local/share/bergamot/models/browsermt";
  std::string folder = "ende.student.tiny11";

  auto prefix_browsermt = [&](const std::string &relative_path) {
    std::string path =
        home + "/" + browsermt + "/" + folder + "/" + relative_path;
    // std::cout << path << "\n";
    return path;
  };

  Record<std::string> path{
      .model = prefix_browsermt("model.intgemm.alphas.bin"),  //
      .vocabulary = prefix_browsermt("vocab.deen.spm"),       //
      .shortlist = prefix_browsermt("lex.s2t.bin")            //
  };

  Record<io::MmapFile> mmap{
      .model = io::MmapFile(path.model),            //
      .vocabulary = io::MmapFile(path.vocabulary),  //
      .shortlist = io::MmapFile(path.shortlist),    //
  };

  Record<View> view{
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };

  Config config;
  Translator translator(config, view.model, view.shortlist, view.vocabulary);
  std::string source = "1 2\n1 2 3\n";
  slimt::Options opts;
  Response response = translator.translate(source, opts);
  fprintf(stdout, "%s\n", response.target.text.c_str());
}

void ShortlistGen() {
  std::string home = std::getenv("HOME");
  std::string browsermt = ".local/share/bergamot/models/browsermt";
  std::string folder = "ende.student.tiny11";

  auto prefix_browsermt = [&](const std::string &relative_path) {
    std::string path =
        home + "/" + browsermt + "/" + folder + "/" + relative_path;
    // std::cout << path << "\n";
    return path;
  };
  std::string vocab_path = prefix_browsermt("vocab.deen.spm");
  std::string shortlist_path = prefix_browsermt("lex.s2t.bin");

  Vocabulary vocab(vocab_path);
  Vocabulary &source = vocab;
  Vocabulary &target = vocab;

  // Load ShortlistGenerator
  io::MmapFile shortlist_file(shortlist_path);
  ShortlistGenerator shortlist_generator(            //
      shortlist_file.data(), shortlist_file.size(),  //
      source, target                                 //
  );

  std::string line = "May I try the shortlist on, please?";
  auto [words, views] = vocab.encode(line, /*add_eos=*/true);
  Shortlist shortlist = shortlist_generator.generate(words);

  const auto &likely_target_words = shortlist.words();
  auto [decoded, dviews] = vocab.decode(likely_target_words);
  for (size_t i = 0; i < likely_target_words.size(); i++) {
    std::cout << "[" << dviews[i] << ": " << likely_target_words[i] << "] ";
  }

  // std::cout << decoded << "\n";
}

}  // namespace slimt

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <test-name>\n";
    std::exit(EXIT_FAILURE);
  }

// clang-format off
#define TEST_ENTRY(fn_name) {#fn_name, &slimt::fn_name}
  // clang-format on

  using Test = void (*)();
  std::unordered_map<std::string, Test> tests({
      TEST_ENTRY(load),                  //
      TEST_ENTRY(integration),           //
      TEST_ENTRY(RowsNodeOp),            //
      TEST_ENTRY(ScalarMultNodeOp),      //
      TEST_ENTRY(DotBatchedNodeOp),      //
      TEST_ENTRY(TransposeNodeOp),       //
      TEST_ENTRY(LayerNormalizationOp),  //
#ifdef SLIMT_HAS_INTGEMM
      TEST_ENTRY(AffineIntgemm),  //
#endif
      TEST_ENTRY(ShortlistGen)  //
  });

  // std::cout << "slimt test\n";
  std::string test = argv[1];

  auto query = tests.find(test);
  if (query != tests.end()) {
    auto name = query->first;
    auto fn = query->second;
    try {
      std::cout << "Running test [" << name << "] ...";
      fn();
      std::cout << " [success]\n";
    } catch (...) {
      std::cout << " [fail]\n";
      throw;
    }
  } else if (test == "all") {
    std::vector<std::string> failed;
    for (auto &named_test : tests) {
      auto name = named_test.first;
      auto fn = named_test.second;
      try {
        std::cout << "Running test ... ";
        fn();
        std::cout << "[success] [" << name << "]\n";
      } catch (const std::exception &exception) {
        std::cout << " [fail] [" << name << "]\n";
        throw;
      }
    }
  } else {
    std::cerr << "Unknown test " << test << "\n";
    std::exit(EXIT_FAILURE);
  }
  return 0;
}
