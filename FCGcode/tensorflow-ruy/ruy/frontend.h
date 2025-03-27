/* Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Implementation of MulFrontEnd, the front-end part of ruy.
// This is what the ruy::Mul entry point calls, and this ends in a call to
// TrMul, at which point we enter the middle-end.
// The front-end work includes parameter validation (Validate), detemplatization
// and resolution of the specific code path to take (CreateTrMulParams), and
// any additional logic best done upfront before entering the middle-end
// (e.g. HandlePrepackedCaching).
// The call to CreateTrMulParams is an important watershed in this code's
// structure: code before it needs to be templatized like the ruy::Mul entry
// point, code after it is un-templatized.

#ifndef RUY_RUY_FRONTEND_H_
#define RUY_RUY_FRONTEND_H_

#include "ruy/create_trmul_params.h"
#include "ruy/ctx.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/trace.h"
#include "ruy/trmul_params.h"
#include "ruy/validate.h"
#include <iostream>
namespace ruy {

// The first half of front-end work, up to the point where we have TrMulParams.
// In other words, this is the part of the front-end work that needs to be
// templatized like the entry point, and that performs the initial work that
// requires this templatization, and the de-templatization. The output of this
// function is the TrMulParams, which contain enough information to allow the
// un-templatized code to take over from there.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename AccumScalar, typename DstScalar>
void MulFrontEndUpToCreateTrMulParams(
    const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
    const Mat<DstScalar>& dst,
    const MulParams<AccumScalar, DstScalar>& mul_params, Ctx* ctx,
    TrMulParams* params) {//
  RUY_TRACE_SCOPE;
  static_assert(CompiledPaths != Path::kNone, "Must compile at least one Path");//静态断言，确保 CompiledPaths 不是 Path::kNone，要求至少编译一个路径。
  static_assert(
      (CompiledPaths & ~kAllPathsIncludingInternalVariants) == Path::kNone,
      "CompiledPaths must be a subset of "
      "ruy::kAllPathsIncludingInternalVariants");//静态断言，确保 CompiledPaths 是 ruy::kAllPathsIncludingInternalVariants 的子集。
//std::cout<<"fro 58"<<std::endl;
  // Perform validation of parameters early so that failures are easier to map
  // to user errors. In particular, perform this validation before the
  // transposition.
  Validate(lhs, rhs, dst);//验证矩阵参数，确保它们满足执行矩阵乘法的要求
//std::cout<<"fro 63"<<std::endl;
  // De-templatize this Mul call by creating a TrMulParams structure.
  // This is also where the specific kernel and pack code paths corresponding to
  // `the_path` are selected, among all the code paths in `CompiledPaths`, and
  // recorded as function pointers in the TrMulParams.
  // The Transpose(lhs) here is where we switch from 'Mul' to 'TrMul'.
  CreateTrMulParams<CompiledPaths>(Transpose(lhs), rhs, dst, mul_params, ctx,
                                   params);//通过调用 CreateTrMulParams 函数来创建 TrMulParams 参数，这个过程中还进行了矩阵转置操作 (Transpose(lhs))。在这一步，选择了与 CompiledPaths 中指定的路径（the_path）对应的特定内核和打包代码路径，并将它们记录为 TrMulParams 中的函数指针
}

// The second part of the front-end work, starting from where we have freshly
// created TrMulParams, performing any remaining front-end work and entering the
// middle-end.
void MulFrontEndFromTrMulParams(Ctx* ctx, TrMulParams* params,int th);

// Top-level function orchestrating the two halves of front-end work:
// before and after we have detemplatized the call by creating TrMulParams.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename AccumScalar, typename DstScalar>
void MulFrontEnd(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                 const MulParams<AccumScalar, DstScalar>& mul_params, Ctx* ctx,
                 Mat<DstScalar>* dst,int th) {
  
  RUY_TRACE_SCOPE;
  profiler::ScopeLabel mul_label("Mul");
  profiler::ScopeLabel shape_specific_label("matmul shape: %dx%dx%d",
                                            lhs.layout.rows, lhs.layout.cols,
                                            rhs.layout.cols);//创建性能分析标签，用于记录性能数据，包括 "Mul" 和矩阵的形状信息
  ctx->clear_performance_advisories();//清除性能建议
  TrMulParams params;//创建用于矩阵乘法的参数对象
  MulFrontEndUpToCreateTrMulParams<CompiledPaths>(lhs, rhs, *dst, mul_params,
                                                  ctx, &params);//执行矩阵乘法的前端操作，创建参数对象 params,参数的验证和创建 TrMulParams 的操作，但并未执行实际的矩阵乘法计算。
//std::cout<<"front"<<std::endl;
  MulFrontEndFromTrMulParams(ctx, &params,th);//执行剩余的矩阵乘法前端操作，将参数对象传递给它。
}

}  // namespace ruy

#endif  // RUY_RUY_FRONTEND_H_
