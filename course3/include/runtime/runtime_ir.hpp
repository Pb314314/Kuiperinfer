#include "ir.h"
#include "runtime/runtime_operand.hpp"
#include "runtime_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace kuiper_infer {

/// 计算图结构，由多个计算节点和节点之间的数据流图组成
class RuntimeGraph {
public:
  /**
   * initialize runtime graph
   * @param param_path 计算图的结构文件
   * @param bin_path 计算图中的权重文件
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * set weight file path
   * @param bin_path 权重文件路径
   */
  void set_bin_path(const std::string &bin_path);

  /**
   * set parameter file path
   * @param param_path  结构文件路径
   */
  void set_param_path(const std::string &param_path);

  /**
   * return the parameter file path
   * @return 返回结构文件
   */
  const std::string &param_path() const;

  /**
   * return the weight file path
   * @return 返回权重文件
   */
  const std::string &bin_path() const;

  /**
   * initialize the graph
   * @return 是否初始化成功
   */
  bool Init();

  const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

private:
  /**
   * 初始化kuiper infer计算图节点中的输入操作数
   * @param inputs pnnx中的输入操作数
   * @param runtime_operator 计算图节点
   */
  static void InitGraphOperatorsInput(const std::vector<pnnx::Operand *> &inputs,const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图节点中的输出操作数
   * @param outputs pnnx中的输出操作数
   * @param runtime_operator 计算图节点
   */
  static void InitGraphOperatorsOutput(const std::vector<pnnx::Operand *> &outputs,const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点属性
   * @param attrs pnnx中的节点属性
   * @param runtime_operator 计算图节点
   */
  static void
  InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,const std::shared_ptr<RuntimeOperator> &runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点参数
   * @param params pnnx中的参数属性
   * @param runtime_operator 计算图节点
   */
  static void
  InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,const std::shared_ptr<RuntimeOperator> &runtime_operator);

public:
private:
  std::string input_name_;  // input node name
  std::string output_name_; // output node name
  std::string param_path_;  // parameter file of the graph
  std::string bin_path_;    // weight file of the graph

  std::vector<std::shared_ptr<RuntimeOperator>> operators_;   // all the operators of the graph
  std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;    // operator mapping of the graph

  std::unique_ptr<pnnx::Graph> graph_;    // pnnx::graph
};

} // namespace kuiper_infer