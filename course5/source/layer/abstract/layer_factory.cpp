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

// Created by fss on 22-11-17.

#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
  // register creator to CreateRegistry
void LayerRegisterer::RegisterCreator(const std::string& layer_type, const Creator& creator) {
  CHECK(creator != nullptr);
  CreateRegistry& registry = Registry();
  // check not already inserted
  CHECK_EQ(registry.count(layer_type), 0) << "Layer type: " << layer_type << " has already registered!";
  // insert [string, Registor function pointer]
  registry.insert({layer_type, creator});
}
// Return the global unique Registry, which is mapping<string, Creator>
LayerRegisterer::CreateRegistry& LayerRegisterer::Registry() {
  // this is static pointer. static variable only initialize once. 
  // static variable is thread safe(if multiple thread call this function, this static variable only initialize once)
  // local static variable, initialize at first use
  // for local static variable, its memory is allocated when program starts(variable isn't constructed yet)
  // Initialization of local static variable occurs the first used.
  static CreateRegistry* kRegistry = new CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
// input operator, use operator->type to find creator function in Registry
// Create and  initialize the layer use creator and return layer
std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<RuntimeOperator>& op) {
  CreateRegistry& registry = Registry();
  const std::string& layer_type = op->type;
  LOG_IF(FATAL, registry.count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;
  const auto& creator = registry.find(layer_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer;// 空的layer
  const auto& status = creator(op, layer);
  LOG_IF(FATAL, status != ParseParameterAttrStatus::kParameterAttrParseSuccess) << "Create the layer: " << layer_type << " failed, error code: " << int(status);
  return layer;
}

std::vector<std::string> LayerRegisterer::layer_types() {
  std::vector<std::string> layer_types;
  static CreateRegistry& registry = Registry();
  for (const auto& [layer_type, _] : registry) {
    layer_types.push_back(layer_type);
  }
  return layer_types;
}
}  // namespace kuiper_infer
