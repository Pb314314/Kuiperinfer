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

// Created by fss on 22-11-15.
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {

const std::vector<std::shared_ptr<Tensor<float>>>& Layer::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

const std::vector<std::shared_ptr<Tensor<float>>>& Layer::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<float>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

// use inputs tensor to compute and save output tensor in outputs
InferStatus Layer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

// Get input tensor vector and output tensor, call foward(input, output)
InferStatus Layer::Forward() {
  LOG_IF(FATAL, this->runtime_operator_.expired()) << "Runtime operator is expired or nullptr";
  const auto& runtime_operator = this->runtime_operator_.lock();
  // Get the input operand of current operator
  const std::vector<std::shared_ptr<RuntimeOperand>>& input_operand_datas = runtime_operator->input_operands_seq;
  // Get input Tensor pushback into layer_input_datas
  std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
  for (const auto& input_operand_data : input_operand_datas) {
    for (const auto& input_data : input_operand_data->datas) {
      layer_input_datas.push_back(input_data);
    }
  }
  // get the output operand of current operator
  const std::shared_ptr<RuntimeOperand>& output_operand_datas = runtime_operator->output_operands;

  CHECK(!layer_input_datas.empty()) << runtime_operator->name << " Layer input data is empty";
  CHECK(output_operand_datas != nullptr && !output_operand_datas->datas.empty()) << "Layer output data is empty";
  
  // use input vector<tensor> to compute the output tensor and save in output_operand_datas->datas
  InferStatus status = runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
  return status;
}

// bind runtime_operator with layer
void Layer::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}

}  // namespace kuiper_infer