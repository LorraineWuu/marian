#pragma once

#include <memory>
#include <iostream>
#include <thread>
#include <cublas_v2.h>

#include "common/keywords.h"
#include "tensors/tensor.h"
#include "graph/chainable.h"


namespace marian {

class Node : public Chainable<Tensor>,
             public keywords::Keywords,
             public std::enable_shared_from_this<Node> {

  protected:
    size_t id_{0};
    size_t edges_{0};
    bool trainable_{true};
    std::vector<Expr> children_;

    ExpressionGraphPtr graph_{nullptr};
    Shape shape_{1, 1, 1, 1};
    std::string name_{"none"};

    Tensor val_{nullptr};
    Tensor adj_{nullptr};

    bool markedForDebug_{false};
    std::string debugMessage_;

  public:
    template <typename ...Args>
    Node(ExpressionGraphPtr graph, Args ...args)
     : Keywords(args...),
       graph_(graph),
       shape_(Get(keywords::shape, {1, 1, 1, 1}))
    {}

    virtual ~Node() {}

    virtual float scalar();

    virtual NodeOps forwardOps() { return {}; };
    virtual NodeOps backwardOps() { return {}; };

    virtual void runForward(const NodeOps& ops) {
      for(auto&& op : ops)
        op();
    }

    virtual void runBackward(const NodeOps& ops) {
      size_t i = 0;
      for(auto&& op : ops)
        if(children()[i++]->trainable())
          op();
    }

    virtual void forward(bool fake=false) {
      if(!fake)
        runForward(forwardOps());
    }

    virtual void backward(bool fake=false) {
      if(!fake)
        runBackward(backwardOps());
    }


    virtual bool trainable() {
      return trainable_;
    }

    virtual void setTrainable(bool trainable) {
      trainable_ = trainable;
    }

    virtual void setId(size_t id) {
      id_ = id;
    }

    virtual size_t getId() {
      return id_;
    }

    virtual void increaseEdges(size_t edges = 1) { edges_ += edges; };
    virtual void decreaseEdges(size_t edges = 1) { edges_ -= edges; };
    virtual size_t edges() { return edges_; };


    virtual ExpressionGraphPtr graph() {
      return graph_;
    }

    virtual void debug(const std::string& message) {
      debugMessage_ = message;
      markedForDebug_ = true;
    }

    virtual bool marked_for_debug() { return markedForDebug_; }
    virtual const std::string& debug_message() { return debugMessage_; }

    virtual size_t allocate(bool);

    virtual void free(bool);

    virtual void init(bool) {};

    virtual void init_dependent(bool);

    virtual void set_zero_adjoint(bool);

    virtual Tensor& val()  {
      return val_;
    };

    virtual Tensor& grad() {
      return adj_;
    };

    virtual const Shape& shape() {
      return shape_;
    }

    void set_name(const std::string& name) {
      name_ = name;
    }

    const std::string &name() const { return name_; }

    virtual const std::string form() {
      return "box";
    }

    virtual const std::string color() {
      return "orange";
    }

    virtual const std::string label() {
      std::stringstream label;
      label << "<" << type();
      if(name_ != "none") {
        label << "<br/>" << "\"" << name_ << "\"";
      }
      label << " (" << getId() << "/" << trainable() << ")>";
      return label.str();
    }

    virtual std::string graphviz() {
      std::stringstream ss;
      ss << "\"" << this << "\" [shape=\"" << form() << "\", label=" << label()
        << ", style=\"filled\", fillcolor=\"" << color() << "\"]" << std::endl;
      for(auto&& child : children())
        ss << "\"" << child << "\" -> \"" << this << "\"" << std::endl;
      ss << std::endl;
      return ss.str();
    }

    virtual std::vector<Expr>& children() {
      return children_;
    }

    cublasHandle_t getCublasHandle();
};

struct NaryNodeOp : public Node {
  size_t hash_{0};
  std::vector<Expr> children_;

  template <typename ...Args>
  NaryNodeOp(const std::vector<Expr>& nodes, Args ...args)
   : Node(nodes.front()->graph(),
      keywords::shape=keywords::Get(keywords::shape, nodes.front()->shape(), args...),
      args...), children_(nodes)
  {
    setTrainable(std::any_of(nodes.begin(), nodes.end(),
                             [](Expr a) { return a->trainable(); } ));
    remove_children_from_top_nodes();
  }

  ~NaryNodeOp() {}

  std::vector<Expr>& children() {
    return children_;
  }

  virtual size_t hash() {
    if(!hash_) {
      std::size_t seed = boost::hash<std::string>()(name());
      boost::hash_combine(seed, type());
      for(auto child : children())
        boost::hash_combine(seed, child->hash());
      hash_ = seed;
    }
    return hash_;
  }

  void remove_children_from_top_nodes();
};

}
