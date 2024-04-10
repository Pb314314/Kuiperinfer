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

  // information of all input operands
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;  // input operands mapping 
  std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;  // vector of input operands
  
  // information of one output operand and several output operators
  std::vector<std::string> output_names;  // all output operators' names
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;  // output operators mapping 
  std::shared_ptr<RuntimeOperand> output_operands;  // one output operand of current operator

  // parameter info: contain parameter information, pb modify this params to use shared_pointer
  std::map<std::string, std::shared_ptr<RuntimeParameter>> params;
  // attribute info: contain all the weight information 
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;  // attribute info
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
