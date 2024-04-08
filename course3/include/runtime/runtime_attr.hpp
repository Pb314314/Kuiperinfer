//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#include <glog/logging.h>
#include <vector>
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace kuiper_infer {

struct RuntimeAttribute {
  std::vector<char> weight_data;  // store the weight value of operand
  std::vector<int> shape;         // store the shape of operand
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;  /// value type of operand

  /**
   * return weight vector from the attribute
   * @tparam T weight type
   * @return vector of weight
   */
  template <class T>  // declare of template function, every template function need to have this;
  std::vector<T> get(bool need_clear_weight = true);

  /**
   * clear weight
   */
  void ClearWeight();
};

template <class T>
std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeUnknown);
  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {  // load float value into vector
      const bool is_float = std::is_same<T, float>::value;  // true if(T == float) else false;
      CHECK_EQ(is_float, true);
      const uint32_t float_size = sizeof(float);
      CHECK_EQ(weight_data.size() % float_size, 0);
      for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
        float weight = *((float*)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight data type: " << int(type);
    }
  }
  if (need_clear_weight) {
    this->ClearWeight();
  }
  return weights;
}

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
