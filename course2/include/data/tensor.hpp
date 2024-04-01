//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <armadillo>
#include <memory>
#include <vector>

namespace kuiper_infer {
template <typename T = float>   // template declaration.
class Tensor {};// Gerneral class, can have nothing inside.

template <>
class Tensor<uint8_t> {}; // A calss that use uint8_t, not implemented yet

// Template specialization
template <>
class Tensor<float> {   // A class that use float.
 public:
  explicit Tensor() = default;

  /**
   * Constructor
   * @param channels 
   * @param rows 
   * @param cols 
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /**
   * Constructor for one-dimension
   * @param size 
   */
  explicit Tensor(uint32_t size);

  /**
   * 创建一个二维向量
   * @param rows 二维向量的高度
   * @param cols 二维向量的宽度
   */
  explicit Tensor(uint32_t rows, uint32_t cols);
  // Use explicit to avoid implicit call of the constructor.
  // e.g. Tensor obj = {3,3};

  /**
   * Create tensor use shape vector.
   * @param shapes 
   */
  explicit Tensor(const std::vector<uint32_t>& shapes);

  Tensor(const Tensor& tensor);

  Tensor(Tensor&& tensor) noexcept;

  Tensor<float>& operator=(Tensor&& tensor) noexcept;

  Tensor<float>& operator=(const Tensor& tensor);

  /**
   * 
   * @return number of rows
   */
  uint32_t rows() const;

  /**
   * 
   * @return number of columns
   */
  uint32_t cols() const;

  /**
   * 
   * @return number of channels
   */
  uint32_t channels() const;

  /**
   * 
   * @return number of elements in the tensor
   */
  uint32_t size() const;

  /**
   *  Set data in the tensor.
   * @param data 
   */
  void set_data(const arma::fcube& data);

  /**
   * Whether tensor is empty
   * @return 
   */
  bool empty() const;

  /**
   * Return the data of the tensor in the location of offset
   * @param offset 
   * @return data value at offset location
   */
  float index(uint32_t offset) const;

  /**
   * return the data at the offset location 
   * @param offset 需要访问的位置
   * @return the reference of the data at the offset location
   */
  float& index(uint32_t offset);

  /**
   * Shape of the tensor
   * @return 张量的尺寸大小
   */
  std::vector<uint32_t> shapes() const;

  /**
   * 张量的实际尺寸大小
   * @return 张量的实际尺寸大小
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * return the data
   * @return reference of the data 
   */
  arma::fcube& data();

  /**
   * 
   * @return return the read only data of the tensor
   */
  const arma::fcube& data() const;

  /**
   * return the data of input channel
   * @param channel 
   * @return the data of input channel
   */
  arma::fmat& slice(uint32_t channel);

  /**
   * 
   * @param channel 
   * @return return the read only data of the input channel
   */
  const arma::fmat& slice(uint32_t channel) const;

  /**
   * 
   * @param channel 
   * @param row 
   * @param col 
   * @return return the data of input location
   */
  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  /**
   * 
   * @param channel 
   * @param row 
   * @param col 
   * @return return the reference of the data at the input location
   */
  float& at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * 
   * @param pads 3-dimension vector of paddding
   * @param padding_value The value to fill the padding
   */
  void Padding(const std::vector<uint32_t>& pads, float padding_value);

  /**
   * initializa the tensor using value.
   * @param value
   */
  void Fill(float value);

  /**
   * 
   * @param values initializa tensor using vector of values.
   */
  void Fill(const std::vector<float>& values, bool row_major = true);

  /**
   * return all data inside tensor
   * @param row_major row major or column major
   * @return vector of data of tensor
   */
  std::vector<float> values(bool row_major = true);

  /**
   * initialize tensor will all ones
   */
  void Ones();

  /**
   * randomly initialize tensor
   */
  void Rand();

  /**
   * print tensor
   */
  void Show();

  /**
   * Reshape the tensor
   * @param shapes new shape of the tensor
   * @param row_major reshape approach
   */
  void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

  /**
   * Flatten the tensor
   */
  void Flatten(bool row_major = false);

  /**
   * Filter the data inside tensor.
   * @param filter filter function
   */
  void Transform(const std::function<float(float)>& filter);

  /**
   * return the row pointer of the data
   * @return return the row pointer of the data
   */
  float* raw_ptr();

  /**
   * Return the row pointer at offset
   * @param offset 
   * @return the row pointer at the offset
   */
  float* raw_ptr(uint32_t offset);

  /**
   * starting pointer of index-th matrix
   * @param index 第index个矩阵
   * @return starting pointer of index-th matrix
   */
  float* matrix_raw_ptr(uint32_t index);

 private:
  std::vector<uint32_t> raw_shapes_;  // raw shape of the tensor
  // 1d raw_shapes_ : {columns}
  // 2d raw_shapes_ : {columns, rows}
  // 3d raw_shapes_ : {columns, rows, channel}
  arma::fcube data_;                  // real data of the tensor
};

using ftensor = Tensor<float>;          // alias for tensor<float>( can use ftensor as a shorthand for Tenser<float>)
using sftensor = std::shared_ptr<Tensor<float>>;    // alias for std::shared_ptr<Tensor<float>>. Simplify code.
// Use aliases make code more concise and readable.
// using or typedef

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_DATA_BLOB_HPP_
