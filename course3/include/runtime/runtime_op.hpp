//
// Created by fss on 22-11-28.
//
#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
// #include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
class Layer;

// operator struct
// contain several input operands, one output operand
// several operators.
struct RuntimeOperator {
  virtual ~RuntimeOperator();

  bool has_forward = false;
  std::string name;      // name of operator
  std::string type;      // type of operator: Convolution, Relu
  std::shared_ptr<Layer> layer;  // layer info

  std::vector<std::string> output_names;  /// output name
  std::shared_ptr<RuntimeOperand> output_operands;  // output operands of current operator
  // input operands mapping 
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;  
  // a vector of input operands
  std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;  // vector of input operands
  // a mapping of output operators

  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;  // output operators mapping 

  // parameter info: contain parameter information
  std::map<std::string, RuntimeParameter*> params;  
  // attribute info: contain all the weight information 
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;  // attribute info
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
