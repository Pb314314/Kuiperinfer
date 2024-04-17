// Created by Bo Pang on 4/16/2024

#ifndef KUIPER_INFER_SOURCE_LAYER_BINOCULAR_SIGMOID_HPP_
#define KUIPER_INFER_SOURCE_LAYER_BINOCULAR_SIGMOID_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer{
class SigmoidLayer : public NonParamLayer{
    public:
    SigmoidLayer(): NonParamLayer("Sigmoid"){}    
    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& input, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& sigmoid_layer);
};
}


#endif