#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace kuiper_infer {
    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
            : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

    void RuntimeGraph::set_bin_path(const std::string &bin_path) {
        this->bin_path_ = bin_path;
    }

    void RuntimeGraph::set_param_path(const std::string &param_path) {
        this->param_path_ = param_path;
    }

    const std::string &RuntimeGraph::param_path() const {
        return this->param_path_;
    }

    const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

    bool RuntimeGraph::Init() {
        // need to set bin_path and param_path before init
        if (this->bin_path_.empty() || this->param_path_.empty()) {
            LOG(ERROR) << "The bin path or param path is empty";
            return false;
        }

        this->graph_ = std::make_unique<pnnx::Graph>();
        int load_result = this->graph_->load(param_path_, bin_path_);
        if (load_result != 0) {
            LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
                       << " " << bin_path_;
            return false;
        }

        std::vector<pnnx::Operator *> operators = this->graph_->ops;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        this->operators_.clear();
        this->operators_maps_.clear();
        for (const pnnx::Operator *op: operators) {         // every operator in graph operators.
            if (!op) {
                LOG(ERROR) << "Meet the empty node";
                continue;
            } else {
                std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();    // initialize a new run_time operator
                // initialize operator name
                runtime_operator->name = op->name;
                runtime_operator->type = op->type;

                // initialize the input operand of current operator, using op->inputs
                const std::vector<pnnx::Operand *> &inputs = op->inputs;        // all input operand for current operator
                if (!inputs.empty()) {
                    InitGraphOperatorsInput(inputs, runtime_operator);          // set input operand vector and mapping for current Runtime_operator 
                }
                // Initialize the output operator of current operator, using op->outputs
                const std::vector<pnnx::Operand *> &outputs = op->outputs;      // all output operand for current operator
                if (!outputs.empty()) {
                    InitGraphOperatorsOutput(outputs, runtime_operator);
                }
                // Just set all operators' name vector, not set the output_operators vector, not set the output_operands

                // Initialize the attribute of current operator, using op->attrs
                const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;// all attribute for current operator
                if (!attrs.empty()) {
                    InitGraphAttrs(attrs, runtime_operator);                    // set attribute mapping for all attribute of current operator ({name, runtime_attribute})
                }

                // Initialize the parameter of current operator, using op->params
                const std::map<std::string, pnnx::Parameter> &params = op->params;// all parameter for current operator
                if (!params.empty()) {
                    InitGraphParams(params, runtime_operator);                  // set parameter mapping for all parameter of current operator({name, runtime_parameter})
                }
                this->operators_.push_back(runtime_operator);
                this->operators_maps_.insert({runtime_operator->name, runtime_operator});
            }
        }
        return true;
    }

    // initialize the RuntimeOperand input operators
    void RuntimeGraph::InitGraphOperatorsInput(const std::vector<pnnx::Operand *> &inputs, const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *input: inputs) {
            if (!input) {
                continue;
            }
            const pnnx::Operator *producer = input->producer;       
            std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
            runtime_operand->name = producer->name;
            runtime_operand->shapes = input->shape;

            switch (input->type) {
                case 1: {
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                }
                case 0: {
                    runtime_operand->type = RuntimeDataType::kTypeUnknown;
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
                }
            }
            runtime_operator->input_operands.insert({producer->name, runtime_operand});
            runtime_operator->input_operands_seq.push_back(runtime_operand);
        }
    }
    // Initialize the output operators
    void RuntimeGraph::InitGraphOperatorsOutput(const std::vector<pnnx::Operand *> &outputs,const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *output: outputs) {    // every output operand
            if (!output) {
                continue;
            }
            const auto &consumers = output->consumers;  // every output operand have serveral operator consumers
            for (const auto &c: consumers) {
                runtime_operator->output_names.push_back(c->name);      // just push output operator consumers' names into output_names
            }
        }
    }

    void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for(const auto & [name, parameter] : params){
            const int type = parameter.type;
            switch (type)
            {
            case int(RuntimeParameterType::kParameterUnknown):{
                std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterBool):{     // 
                std::shared_ptr<RuntimeParameterBool> runtime_parameter = std::make_shared<RuntimeParameterBool>();
                runtime_parameter->value = parameter.b;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterInt): {
                std::shared_ptr<RuntimeParameterInt> runtime_parameter = std::make_shared<RuntimeParameterInt>();
                runtime_parameter->value = parameter.i;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
                }
            case int(RuntimeParameterType::kParameterFloat): {
                std::shared_ptr<RuntimeParameterFloat> runtime_parameter = std::make_shared<RuntimeParameterFloat>();
                runtime_parameter->value = parameter.f;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterString): {
                std::shared_ptr<RuntimeParameterString> runtime_parameter = std::make_shared<RuntimeParameterString>();
                runtime_parameter->value = parameter.s;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterIntArray): {
                std::shared_ptr<RuntimeParameterIntArray> runtime_parameter = std::make_shared<RuntimeParameterIntArray>();
                runtime_parameter->value = parameter.ai;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterFloatArray): {
                std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter = std::make_shared<RuntimeParameterFloatArray>();
                runtime_parameter->value = parameter.af;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterStringArray): {
                std::shared_ptr<RuntimeParameterStringArray> runtime_parameter = std::make_shared<RuntimeParameterStringArray>();
                runtime_parameter->value = parameter.as;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            default: {
                LOG(FATAL) << "Unknown parameter type: " << type;
            }
            }
        /*
        for (const auto &[name, parameter]: params) {
            const int type = parameter.type;
            switch (type) {     // assign value based on data type
                case int(RuntimeParameterType::kParameterUnknown): {
                    RuntimeParameter *runtime_parameter = new RuntimeParameter;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterBool): {
                    RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
                    runtime_parameter->value = parameter.b;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterInt): {
                    RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
                    runtime_parameter->value = parameter.i;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterFloat): {
                    RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
                    runtime_parameter->value = parameter.f;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterString): {
                    RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
                    runtime_parameter->value = parameter.s;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterIntArray): {
                    RuntimeParameterIntArray *runtime_parameter = new RuntimeParameterIntArray;
                    runtime_parameter->value = parameter.ai;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterFloatArray): {
                    RuntimeParameterFloatArray *runtime_parameter = new RuntimeParameterFloatArray;
                    runtime_parameter->value = parameter.af;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                case int(RuntimeParameterType::kParameterStringArray): {
                    RuntimeParameterStringArray *runtime_parameter =
                            new RuntimeParameterStringArray;
                    runtime_parameter->value = parameter.as;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown parameter type: " << type;
                }
            }
        }*/
        }
    }

    void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, attr]: attrs) {
            switch (attr.type) {
                case 1: {       // float type data
                    std::shared_ptr<RuntimeAttribute> runtime_attribute =std::make_shared<RuntimeAttribute>();
                    runtime_attribute->type = RuntimeDataType::kTypeFloat32;
                    runtime_attribute->weight_data = attr.data;
                    runtime_attribute->shape = attr.shape;
                    runtime_operator->attribute.insert({name, runtime_attribute});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
            }
        }
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> & RuntimeGraph::operators() const {
        return this->operators_;
    }

} // namespace kuiper_infer
