//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace kuiper_infer {

// The way to construct tensor: set dimension and construct fcube use the dimension
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);  // The way to create 3-D matrix in fcube
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);  // {1, 1, 1} 
  // copy elements from shapes to shapes_
  // If input : {2,3}, shapes_ : {1,2,3} set one from begenning
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);    // initialize data shape
  if (channels == 1 && rows == 1) {             // set raw shapes
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}
// Copy constructor
Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}
// Move constructor
Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}
// set data with same dimension input.
void Tensor<float>::set_data(const arma::fcube& data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const { return this->data_.empty(); }

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube& Tensor<float>::data() { return this->data_; }

const arma::fcube& Tensor<float>::data() const { return this->data_; }

arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

// Padding the tensor use pads vector. {up, bottom, left, right}
void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  //using std::cout, std::endl;                            
  CHECK(!this->data_.empty());  // not empty()
  CHECK_EQ(pads.size(), 4);     // 

  uint32_t pad_up = pads.at(0);  // up
  uint32_t pad_bot = pads.at(1);  // bottom
  uint32_t pad_left = pads.at(2);  // left
  uint32_t pad_right = pads.at(3);  // right
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();

  this->data_.insert_rows(0, pad_up);
  for(uint32_t i=0;i<pad_up;i++){
    this->data_.row(i).fill(padding_value);
  }

  this->data_.insert_rows(pad_up + rows, pad_bot);
  for(uint32_t i=0;i<pad_bot;i++){//  why don't need to iterate every channel
    this->data_.row(pad_up + rows + i ).fill(padding_value);
  }

  this->data_.insert_cols(0, pad_left);
  for(uint32_t i=0;i<pad_left;i++){
    this->data_.col(i).fill(padding_value);
  }

  this->data_.insert_cols(pad_left + cols, pad_right);
  for(uint32_t i=0;i<pad_right;i++){
    this->data_.col(cols + pad_left + i).fill(padding_value);
  }
  /* Fail to create bigger padding matrix and insert the original data.
  std::vector<float> num_vec = this->values();
  uint32_t slice = this->rows() * this->cols();
  uint32_t channel = this->channels();
  uint32_t pad_row = this->rows() + pad_up + pad_bot;
  uint32_t pad_col = this->cols() + pad_left + pad_right;

  arma::fcube padding_cube(pad_row, pad_col, this->channels());// {row, column, channel}
  padding_cube.fill(padding_value); // fill padding value to new fcube

  for(uint32_t i=0;i<channel;i++){
    int32_t index = i * slice + pad_left * pad_row + pad_up;
    for(uint32_t j=0;j < cols;j++){
      std::copy(num_vec.begin()+i*slice, num_vec.begin() + i*slice + index + rows, padding_cube.memptr() + index);
      index += pad_row;
    }
  }
  this->data_ = padding_cube;
  */
}

// Fill the tensor with one value.
void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

// Fill the tensor with the vector of data. Use different approach with row_major and column major.
void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);    // get the i-th channel data
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());//(start pointer, cols, rows)
          // construct a float matrix use read only array
      channel_data = channel_data_t.t();      // transpose the matrix for row major. (armadillo is column major)
    }
  } else {
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  uint32_t fcube_size = this->size();
  this->Reshape({fcube_size}, row_major);
  this->raw_shapes_ = {fcube_size};
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

// transform the tensor use filter function.
void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

// Reshape the tensor. Same size after reshape.
void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  std::vector<float> values;
  if (row_major) {
    values = this->values(true);    // Get vector of all the values of tensor.
  }
  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));  // reshape the tensor, data will be  preserved.
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)}; // set the shape value
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);   
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shapes_ = {shapes.at(0)};
  }

  if (row_major) {    // Refill the data if row major, don't need to refill if column major.
    this->Fill(values, true );    // Fill data using vector of data.
  }
}

float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset) {
  const uint32_t size = this->size();
  CHECK(!this->data_.empty());
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<float> values(this->data_.size());

  if (!row_major) { // derectly copy the data to vector if column major.
  // The way data stored in armadillo is column major.
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {  // for row major.
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::fmat& channel = this->data_.slice(c).t();   // get the channel data and transpose.
      std::copy(channel.begin(), channel.end(), values.begin() + index);  // copy the transposed matrix to get row data.
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  uint32_t offset = index * this->rows() * this->cols();
  CHECK_LE(offset, this->size());
  float* mem_ptr = this->raw_ptr() + offset;    // get the value in index position.
  return mem_ptr;
}
}  // namespace kuiper_infer
