#include "sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();        
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor &input_data = inputs.at(i);              // every input tensor
    const sftensor &output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {     // check input data not empty
      LOG(ERROR) << "The input tensor array in the relu layer has an empty tensor " << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the relu "
                    "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {       
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);     // 3 dim tensor
    CHECK(input == nullptr || !input->empty())
            << "The input tensor array in the relu layer has an empty tensor " << i
            << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR)
          << "The output tensor array in the relu layer has an empty tensor "
          << i << " th";
      output = std::make_shared<Tensor<float>>(input->shapes());        // Create output tensor
      outputs.at(i) = output;
    }
    CHECK(output->shapes() == input->shapes())
            << "The input and output tensor shapes of the relu layer do not match "
            << i << " th";

    // Need to transfer every input number to sigmoid result;
    for (uint32_t j = 0; j < input->size(); ++j) {          // compute every float in one tensor
        float value = input->index(j);
        output->index(j) = 1.0f / (1.0f + expf(-value));    // use expf to compute the sigmoid result;
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SigmoidLayer::GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& sigmoid_layer){
    CHECK(op != nullptr) << "Sigmoid operator is nullptr";
    sigmoid_layer = std::make_shared<SigmoidLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSigmoidInstance("nn.Sigmoid", SigmoidLayer::GetInstance);
}