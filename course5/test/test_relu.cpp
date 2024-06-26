//
// Created by fss on 23-6-25.
//
#include "layer/abstract/layer_factory.hpp"
#include <gtest/gtest.h>
using namespace kuiper_infer;

static LayerRegisterer::CreateRegistry *RegistryGlobal() {
  static LayerRegisterer::CreateRegistry *kRegistry = new LayerRegisterer::CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return kRegistry;
}
// Understand this: use singleton implementation to build global unique registry 
TEST(test_registry, registry1) {
  using namespace kuiper_infer;
  LayerRegisterer::CreateRegistry *registry1 = RegistryGlobal();
  LayerRegisterer::CreateRegistry *registry2 = RegistryGlobal();

  LayerRegisterer::CreateRegistry *registry3 = RegistryGlobal();
  LayerRegisterer::CreateRegistry *registry4 = RegistryGlobal();
  float *a = new float{3};
  float *b = new float{4};
  ASSERT_EQ(registry1, registry2);
}

ParseParameterAttrStatus MyTestCreator(const std::shared_ptr<RuntimeOperator> &op,std::shared_ptr<Layer> &layer) {
  layer = std::make_shared<Layer>("test_layer");
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

TEST(test_registry, registry2) {
  using namespace kuiper_infer;
  LayerRegisterer::CreateRegistry registry1 = LayerRegisterer::Registry();
  LayerRegisterer::CreateRegistry registry2 = LayerRegisterer::Registry();
  ASSERT_EQ(registry1, registry2);
  // register creator to Registry
  LayerRegisterer::RegisterCreator("test_type", MyTestCreator);
  LayerRegisterer::CreateRegistry registry3 = LayerRegisterer::Registry();
  // Change to 3, already register nn.Relu and nn.Sigmoid
  ASSERT_EQ(registry3.size(), 3);
  ASSERT_NE(registry3.find("test_type"), registry3.end());
}

TEST(test_registry, create_layer) {
  // 注册了一个test_type_1算子
  LayerRegisterer::RegisterCreator("test_type_1", MyTestCreator);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test_type_1";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_util) {
  LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test_type_2";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_reluforward) {
  // allocata a RuntimeOperator
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  // set the RuntimeOperator type(consistent with layer name)
  op->type = "nn.ReLU";
  // declare a layer
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  // Use CreateLayer to initialize a layer by get Creator function in Registry mapping 
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
  input_tensor->Rand();
  input_tensor->data() -= 0.5f;

  LOG(INFO) << input_tensor->data();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;    // set the first element of input vector.

  // directly use layer->Forward(input_t, output_t) to compute the outputs tensor
  layer->Forward(inputs, outputs);

  for (const auto &output : outputs) {
    output->Show();
  }
}

/*
ParseParameterAttrStatus MyTestCreator(const std::shared_ptr<RuntimeOperator> &op,std::shared_ptr<Layer> &layer) {
  layer = std::make_shared<Layer>("test_layer");
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}*/

TEST(test_registry, pb_test) {
  // allocata a RuntimeOperator

  // Register Op
  LayerRegisterer::CreateRegistry Registry = LayerRegisterer::Registry();
  LayerRegisterer::RegisterCreator("operator3", MyTestCreator);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "operator3";
  // declare a layer
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  // Use CreateLayer to initialize a layer by get Creator function in Registry mapping 
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
  std::cout << layer->layer_name() << std::endl;

  for(const auto [layer_name, _] : Registry){
    std::cout << layer_name <<std::endl;
  }
}