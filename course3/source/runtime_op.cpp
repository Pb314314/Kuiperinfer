//
// Created by fss on 23-2-27.
//
#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer  {
RuntimeOperator::~RuntimeOperator() {
  // Pb delete this, modify second pair element in params to smart pointer
  /*
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }*/
}

}  // namespace kuiper_infer
