// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
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

#include "node_operators.h"
#include "expression_graph.h"

namespace marian {

  size_t ConstantNode::allocate(bool fake) {
    // @TODO params
    size_t elements = 0;
    if(!val_) {
      graph()->tensor(val_, shape_, fake);
      elements = val_->shape().elements();
    }
    return elements;
  }

  void ConstantNode::init(bool fake) {
    if(!initialized_ && !fake) {
      init_(val_);
      initialized_ = true;
    }
  }

  size_t ParamNode::allocate(bool fake) {
    // @TODO params
    size_t elements = 0;
    if(!val_) {
      graph()->tensor(val_, shape_, fake);
      elements = val_->shape().elements();
    }
    return elements;
  }

  void ParamNode::init(bool fake) {
    if(!initialized_ && !fake) {
      //std::cerr << "Initializing parameter " << name() << std::endl;
      init_(val_);
      initialized_ = true;
    }
  }

}
