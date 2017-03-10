#include "graph/expression_graph.h"
#include "graph/node.h"
#include "kernels/tensor_operators.h"

namespace marian {

size_t Node::allocate(bool fake) {
  size_t elements = 0;
  if(!val_) {
    graph_->tensor(val_, shape_, fake);
    elements = val_->shape().elements();
  }
  return elements;
}

void Node::free(bool fake) {
  if(val_)
    graph_->free(val_, fake);
  if(adj_)
    graph_->free(adj_, fake);
}

void Node::init_dependent(bool fake) {
  if(!adj_) {
    graph_->tensor(adj_, shape_, fake);
    if(!fake)
      adj_->set(1);
  }
}

void Node::set_zero_adjoint(bool fake) {
  if(!adj_) {
    graph_->tensor(adj_, shape_, fake);
    if(!fake)
      adj_->set(0);
  }
}

float Node::scalar() {
  return val_->scalar();
}


cublasHandle_t Node::getCublasHandle() {
  return graph_->getCublasHandle();
}

void NaryNodeOp::remove_children_from_top_nodes() {
  for(auto child : children_)
    graph_->remove_top_node(child);
}

}
