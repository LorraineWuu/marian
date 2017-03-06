#pragma once

// This file is part of the Marian toolkit.

//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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

#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "3rd_party/exception.h"
#include "common/definitions.h"
#include "common/shape.h"

#ifdef CUDNN
#include <cudnn.h>
#endif

namespace marian {

class TensorBase : public std::enable_shared_from_this<TensorBase> {
  private:
    float* data_;
    Shape shape_;
    size_t device_;
#ifdef CUDNN
    cudnnTensorDescriptor_t cudnnDesc_;
#endif

  public:
    TensorBase(float* data, Shape shape, size_t device)
      : data_(data), shape_(shape), device_(device)
    {
#ifdef CUDNN
       cudnnCreateTensorDescriptor(&cudnnDesc_);
       cudnnSetTensor4dDescriptor(cudnnDesc_,
                                  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                  shape_[0], shape_[1],
                                  shape_[2], shape_[3]);
#endif
    }

    ~TensorBase()
    {
#ifdef CUDNN
      cudnnDestroyTensorDescriptor(cudnnDesc_);
#endif
    }

    virtual void reset(float* data) {
      data_ = data;
    }

    virtual float* data() {
      return data_;
    }

    virtual Shape& shape() {
      return shape_;
    }

    virtual size_t size() {
      return shape_.elements();
    }

    virtual float scalar() {
      UTIL_THROW_IF2(size() != 1, "Tensor is not a scalar");
      return get(0);
    }

    size_t getDevice() {
      return device_;
    }

    Tensor subtensor(int offset, int size){
      return Tensor(new TensorBase(data_ + offset, {1, size}, device_ ));
    }

    float get(size_t i);

    void set(size_t i, float value);

    void get(std::vector<float> &v);

    void set(float value);

    void set(const std::vector<float> &v);

    void copyFrom(Tensor);

#ifdef CUDNN
      cudnnTensorDescriptor_t cudnn() {
            return cudnnDesc_;
          }
#endif

    std::string debug();
};

class DeviceGPU {
  private:
    float* data_;
    size_t size_;
    size_t device_;

  public:
    DeviceGPU(size_t device)
    : data_(0), size_(0), device_(device) {
   }

    ~DeviceGPU();

    typedef TensorBase tensor_type;

    void reserve(size_t size);

    float* data() {
      return data_;
    }

    size_t capacity() {
      return size_;
    }

    size_t getDevice() {
      return device_;
    }
};

typedef std::shared_ptr<TensorBase> Tensor;

Tensor operator<<(Tensor t, const std::vector<float>& v);

Tensor operator>>(Tensor t, std::vector<float>& v);

}
