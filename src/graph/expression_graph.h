#pragma once

#include <map>
#include <unordered_set>
#include <fstream>

#include "common/definitions.h"
#include "graph/chainable.h"
#include "graph/parameters.h"
#include "graph/node_operators.h"
#include "data/batch_generator.h"
#include "tensors/tensor_allocator.h"
#include "layers/param_initializers.h"
#include "kernels/dropout.h"
#include "3rd_party/threadpool.h"
#include "3rd_party/cnpy/cnpy.h"

namespace marian {

template <class T, typename ...Args>
Expr Expression(Args&& ... args);

/**
 * @brief Represents a computation graph of expressions, over which algorithmic differentiation may be performed.
 */
class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
  private:

    /** @brief The full list of nodes */

    size_t count_{0};

    std::vector<Expr> nodes_;
    std::vector<std::vector<Expr>> tapes_;
    std::map<Expr, size_t> tapeMap_;

    /** @brief Maps from name to expression node. */
    std::map<std::string, Expr> named_;

    /** @brief List of all input nodes of this expression graph. */
    std::vector<Expr> inputs_;

    /** @brief Contains all nodes with regard to which we want to calculate derivatives */
    std::unordered_set<Expr> topNodes_;

    Parameters params_;
    Ptr<TensorAllocator> tensors_;

    cublasHandle_t cublasHandle_;
    curandGenerator_t curandGenerator_;
    size_t device_{0};

    std::unordered_map<size_t, Expr> hashMap_;

  protected:
    /** @brief Constructs a new expression graph
     * Constructor is protected to force use of New<ExpressionGraph>()
    */
    ExpressionGraph() { }

    // delete copy and move constructors
    ExpressionGraph(const ExpressionGraph&) = delete;
    ExpressionGraph(ExpressionGraph&&) = delete;

    friend Ptr<ExpressionGraph> New<ExpressionGraph>();

  public:

    ~ExpressionGraph() {
      clear();
    }

    void setDevice(size_t device = 0) {
      device_ = device;
      params_.init(device);
      tensors_ = New<TensorAllocator>(device);
      cublasHandle_ = create_handle(device);
      curandGenerator_ = createCurandGenerator(device, 1234);
    }

    cublasHandle_t getCublasHandle() {
      return cublasHandle_;
    }

    curandGenerator_t getCurandGenerator() {
      return curandGenerator_;
    }

    size_t getDevice() {
      return device_;
    }

    void reserveWorkspaceMB(size_t num, bool fake=false) {
      size_t elements = num * 1024 * 1024 / 4 - 1;
      tensors_->reserve(elements, fake);
    }

    size_t reservedWorkspaceMB() {
      return (tensors_->peak() * 4) / (1024 * 1024);
    }

    /**
     * @brief Performs backpropogation on this expression graph.
     *
     * Backpropogation is implemented by performing first the forward pass
     *    and then the backward pass of algorithmic differentiation (AD) on the nodes of the graph.
     *
     */
    void backprop() {
      forward();
      backward();
    }

    size_t forward(bool fake=false) {
      params_.allocateForward(fake);
      return forward(0, fake);
    }

    size_t forward(size_t pos, bool fake=false) {
      // @TODO: check if allocation works properly

      auto it = nodes_.begin() + pos;
      while(it != nodes_.end()) {
        auto v = *it;
        v->allocate(fake);
        v->init(fake);
        v->forward(fake);

        // @TODO: should be done in node
        for(auto&& child : v->children()) {
          v->decreaseEdges(1);
          child->decreaseEdges(1);
        }

        if(v->marked_for_debug() && !fake) {
          std::cerr << "Debug: " << v->debug_message() << std::endl;
          std::cerr << v->val()->debug() << std::endl;
        }
        it++;
      }
      return std::distance(nodes_.begin(), it);
    }

    /**
     * @brief Perform the backward pass of algorithmic differentiation (AD) on this graph.
     *
     * This pass traverses the nodes of this graph in reverse of the order they were created;
     *    as each node is traversed, its <code>set_zero_adjoint()</code> method is called.
     *
     * Once this has been performed for all nodes, this pass again traverses the nodes, again in reverse creation order;
     *    as each node is traversed, its <code>backward()</code> method is called.
     *
     * After this method has successfully completed,
     *    and that all backward pass computations have been performed.
     */
    void backward(bool fake=false) {
      UTIL_THROW_IF2(topNodes_.size() > 1,
        "There are more than one top most node for backward step");

      params_.allocateBackward(fake);
      params_.set_zero_adjoint(fake);

      for(auto&& v : topNodes_)
        v->init_dependent(fake);

      auto it = nodes_.rbegin();
      while(it != nodes_.rend()) {
        auto v = *it;

        for(auto&& child: v->children())
          if(child->trainable())
            child->set_zero_adjoint(fake);
        if(v->trainable())
          v->backward(fake);
        for(auto&& child : v->children()) {
          v->decreaseEdges(1);
          child->decreaseEdges(1);
        }

        if(v->trainable() && v->marked_for_debug() && !fake) {
          std::cerr << "Debug Grad: " << v->debug_message() << std::endl;
          std::cerr << v->grad()->debug() << std::endl;
        }

        // delete unnamed nodes
        if(v->edges() == 0 && v->name() == "none")
          v->free(fake);

        it++;
      }
    }

    /**
     * @brief Returns a string representing this expression graph in <code>graphviz</code> notation.
     *
     * This string can be used by <code>graphviz</code> tools to visualize the expression graph.
     *
     * @return a string representing this expression graph in <code>graphviz</code> notation
     */
    std::string graphviz() {
      std::stringstream ss;
      ss << "digraph ExpressionGraph {" << std::endl;
      //ss << "graph[splines=ortho]" << std::endl;
      ss << "rankdir=LR" << std::endl;

      auto it = nodes_.rbegin();
      while(it != nodes_.rend()) {
        auto v = *it;
        ss << v->graphviz();
        it++;
      }

      ss << "}" << std::endl;
      return ss.str();
    }

    void graphviz(const std::string& filename) {
      std::ofstream dot(filename);
      dot << graphviz();
      dot.close();
    }

    void dump(const std::string& filename) {
      std::cerr << "Saving not yet implemented" << std::endl;
    }

    /*********************************************************/

    /**
     * @brief Constructs a new node representing an input in an expression graph.
     *
     * This method records the input node in a list of input nodes,
     *    but does not attach the new input node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed input node
     */
    template <typename ...Args>
    inline Expr input(Args ...args) {
      auto e = Expression<InputNode>(shared_from_this(), args...);
      inputs_.emplace_back(e);
      return e;
    }

    /**
     * @brief Constructs a new node representing a parameter in an expression graph.
     *
     * This method records the parameter node in a list of parameter nodes,
     *    but does not attach the new parameter node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed parameter node
     */
    template <typename ...Args>
    inline Expr param(const std::string& name,
                      Shape shape,
                      Args ...args) {
      // check first if parameter already exists
      auto p = params_.get(name);
      if(p) {
        // if yes add to tape and return
        add(p);
        return p;
      }

      // if not check if name is not taken by other node
      UTIL_THROW_IF2(get(name),
                     "Non-parameter with name "
                     << name
                     << "already exists");

      // create parameter node (adds to tape)
      p = Expression<ParamNode>(shared_from_this(),
                                keywords::shape=shape,
                                args...);

      // add to list of parameters
      p->set_name(name);
      params_.add(p, name);
      return p;
    }

    /**
     * @brief Constructs a new node representing a constant in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr constant(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(), args...);
    }

    /**
     * @brief Constructs a new node representing a constant (with value 1) in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr ones(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=inits::ones,
                                      args...);
    }

    /**
     * @brief Constructs a new node representing a constant (with value 0) in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr zeros(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=inits::zeros,
                                      args...);
    }

    template <typename ...Args>
    inline Expr dropout(float prob, Shape shape) {
      auto dropoutInit = [prob, this](Tensor t) {
        Dropout(t, prob, getCurandGenerator());
      };

      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=dropoutInit,
                                      keywords::shape=shape);
    }

    /*********************************************************/

    /**
     * @brief Returns the first item in the list with the specified name, if such an item exists.
     *
     * If no item with the specified name is found in the graph, this method throws an exception.
     *
     * @param name Name of the desired expression node
     *
     * @return the first item in the list with the specified name, if such an item exists
     */
    Expr get(const std::string& name) {
      auto e = params_.get(name);
      if(e)
        return e;

      auto it = named_.find(name);
      if(it == named_.end())
        return Expr();
      return it->second;
    }

    /**
     * @brief Gets the list of all parameter nodes of this expression graph
     *
     * @return the list of all parameter nodes of this expression graph
     */
    Parameters& params() {
      return params_;
    }

    /**
     * @brief Inserts an expression node with a specified name into the expression graph.
     *
     * @param e an expression node
     * @param name name of the expression node
     *
     * @return the expression node that was added to the expression graph
     */
    void add_named_node(Expr e, const std::string& name) {
      UTIL_THROW_IF2(params_.get(name) || get(name),
                     "Node names must be unique");

      named_.emplace(name, e);
    }

    Expr add(Expr node) {
      size_t group = 0;

      size_t hash = node->hash();
      auto it = hashMap_.find(hash);
      if(it != hashMap_.end())
        return it->second;

      hashMap_[hash] = node;

      node->setId(count_++);

      for(auto& child: node->children()) {
        group = std::max(group, tapeMap_[child] + 1);
        child->increaseEdges(2);
        node->increaseEdges(2);
      }
      tapeMap_[node] = group;
      if(group >= tapes_.size())
        tapes_.resize(group + 1);
      tapes_[group].push_back(node);
      nodes_.push_back(node);
      topNodes_.insert(node);

      return node;
    }

    void remove_top_node(Expr node) {
      topNodes_.erase(node);
    }

    template <class ...Args>
    void tensor(Tensor& t, Args&&... args) {
      tensors_->allocate(t, args...);
    }

    void free(Tensor& t, bool fake) {
      tensors_->free(t, fake);
    }

    void clear() {
      // clear everything apart from parameters
      count_ = 0;
      nodes_.clear();
      tapes_.clear();
      tapeMap_.clear();

      named_.clear();
      inputs_.clear();
      topNodes_.clear();
      tensors_->clear();
      hashMap_.clear();
    }

    Expr topNode() {
      return nodes_.back();
    }

    void load(const std::string& name) {
      using namespace keywords;

      LOG(info) << "Loading model from " << name;

      auto numpy = cnpy::npz_load(name);

      for(auto it : numpy) {
        auto name = it.first;

        Shape shape;
        if(it.second.shape.size() == 2) {
          shape.set(0, it.second.shape[0]);
          shape.set(1, it.second.shape[1]);
        }
        else if(it.second.shape.size() == 1) {
          shape.set(0, 1);
          shape.set(1, it.second.shape[0]);
        }

        param(name, shape,
              init=inits::from_numpy(it.second));
      }
    }

    void save(const std::string& name) {
      LOG(info) << "Saving model to " << name;

      unsigned shape[2];
      std::string mode = "w";

      cudaSetDevice(getDevice());
      for(auto p : params().getMap()) {
        std::vector<float> v;
        p.second->val() >> v;

        unsigned dim;
        if(p.second->shape()[0] == 1) {
          shape[0] = p.second->shape()[1];
          dim = 1;
        }
        else {
          shape[0] = p.second->shape()[0];
          shape[1] = p.second->shape()[1];
          dim = 2;
        }
        std::string pName = p.first;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
        mode = "a";
      }
    }
};

template <class T, typename ...Args>
Expr Expression(Args&& ... args) {
  // @TODO check hash, if exists do not add and return
  // cached node to minimize calculations
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}

}
