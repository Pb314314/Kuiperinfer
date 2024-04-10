// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 23-2-27.
#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
RuntimeOperator::~RuntimeOperator() {
  for (auto &[_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }
}
// Resize Tensor vector for every input operand of every operator 
void RuntimeOperatorUtils::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  if (operators.empty()) {
    LOG(ERROR) << "Operators for init input shapes is empty!";
    return;
  }
  for (const auto &op : operators) {    // every operator
    if (op->input_operands.empty()) {   // no input operand
      continue;
    } 
    else {
      const std::map<std::string, std::shared_ptr<RuntimeOperand>> &input_operands_map = op->input_operands;
      // 初始化operator的输入空间
      for (const auto &[_, input_operand] : input_operands_map) { // every input operand of current operator
        const auto &type = input_operand->type;
        CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!";
        const auto &input_operand_shape = input_operand->shapes;
        // input tensor data of input operand
        auto &input_datas = input_operand->datas;     // std::vector<std::shared_ptr<Tensor<float>>>

        CHECK(!input_operand_shape.empty());
        const int32_t batch = input_operand_shape.at(0);
        CHECK(batch >= 0) << "Dynamic batch size is not supported!";
        CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 3 || input_operand_shape.size() == 4) << "Unsupported tensor shape sizes: " << input_operand_shape.size();

        if (!input_datas.empty()) {CHECK_EQ(input_datas.size(), batch);}    // input datas not empty, check whether size == batch
        else {input_datas.resize(batch);}     // resize vector size to batch size
      }
    }
  }
}

// Allocate memory for operator otput; initialize the output operand in RuntimeOperator, check  operand size if already initialized.
void RuntimeOperatorUtils::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty());
  CHECK(pnnx_operators.size() == operators.size());
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {    // every pnnx operator
    // get the pnnx operands information
    const std::vector<pnnx::Operand *> operands = pnnx_operators[i]->outputs;    
    CHECK(operands.size() <= 1) << "Only support one node one output yet!";
    if (operands.empty()) {continue;}
    CHECK(operands.size() == 1) << "Only support one output in the KuiperInfer";
    // Only support one output operand per operator
    pnnx::Operand *operand = operands.front();    // get the first operand(only one operand in this vector), use this operand to set info to operator's output operand
    
    // get RuntimeOperator information
    const auto &runtime_op = operators.at(i);     // index i, so operator in Runtime_op have same order as pnnx op? 
    CHECK(operand != nullptr) << "Operand output is null";
    const std::vector<int32_t> &operand_shapes = operand->shape;
    const auto &output_tensors = runtime_op->output_operands;     // set output operands for every operator
    const int32_t batch = operand_shapes.at(0);
    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
    CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 3 || operand_shapes.size() == 4)<< "Unsupported shape sizes: " << operand_shapes.size();

    // use pnnx operand to initialize or check RuntimeOperator's output operand.
    if (!output_tensors) {    // uninitialized output tensor
      std::shared_ptr<RuntimeOperand> output_operand = std::make_shared<RuntimeOperand>();// initialize the output operand
      output_operand->shapes = operand_shapes;
      output_operand->type = RuntimeDataType::kTypeFloat32;
      output_operand->name = operand->name + "_output";
      for (int j = 0; j < batch; ++j) {// initialize the output data vector for each batch
        // initialize the operand tensor based on tensor shape
        if (operand_shapes.size() == 4) {   
          sftensor output_tensor = TensorCreate(operand_shapes.at(1), operand_shapes.at(2), operand_shapes.at(3));
          output_operand->datas.push_back(output_tensor);
        } else if (operand_shapes.size() == 2) {
          sftensor output_tensor = TensorCreate((uint32_t) operand_shapes.at(1));
          output_operand->datas.push_back(output_tensor);
        } else {
          // current shape is 3
          sftensor output_tensor = TensorCreate((uint32_t) operand_shapes.at(1), (uint32_t) operand_shapes.at(2));
          output_operand->datas.push_back(output_tensor);
        }
      }
      runtime_op->output_operands = std::move(output_operand);
    } 
    else {    // already initialized output tensor size 
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
      CHECK(output_tensors->shapes == operand_shapes);
      // check every output tensor shape
      for (uint32_t b = 0; b < batch; ++b) {
        sftensor output_tensor = output_tensors->datas[b];
        const std::vector<uint32_t> &tensor_shapes = output_tensor->shapes();
        // check the output tensor shape based on shape
        if (operand_shapes.size() == 4) {   
          if (tensor_shapes.at(0) != operand_shapes.at(1) ||tensor_shapes.at(1) != operand_shapes.at(2) || tensor_shapes.at(2) != operand_shapes.at(3)) {
            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes = std::vector<uint32_t>{
                (uint32_t) operand_shapes.at(1), (uint32_t) operand_shapes.at(2),
                (uint32_t) operand_shapes.at(3)};
            output_tensor->Reshape(target_shapes);
          }
        } else if (operand_shapes.size() == 2) {
          if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) || tensor_shapes.at(2) != 1) {
            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes = std::vector<uint32_t>{(uint32_t) operand_shapes.at(1)};
            output_tensor->Reshape(target_shapes);
          }
        } else {
          // current shape is 3
          if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) || tensor_shapes.at(2) != operand_shapes.at(2)) {
            DLOG(WARNING)
                << "The shape of tensor do not adapting with output operand";
            const auto &target_shapes = std::vector<uint32_t>{
                (uint32_t) operand_shapes.at(1), (uint32_t) operand_shapes.at(2)};
            output_tensor->Reshape(target_shapes);
          }
        }
      }
    }
  }
}

}  // namespace kuiper_infer
