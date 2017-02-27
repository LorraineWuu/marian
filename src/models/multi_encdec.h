#pragma once

#include "data/corpus.h"
#include "training/config.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "common/logging.h"
#include "models/encdec.h"

namespace marian {

class MultiDecoder : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention1_;
    Ptr<GlobalAttention> attention2_;

  public:
    DecoderGNMT(Ptr<Config> options)
     : DecoderBase(options) {}

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         std::pair<Expr> context,
         std::pair<Expr> contextMask,
         bool single) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("normalize");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");
      float dropoutRnn = options_->get<float>("dropout-rnn");
      float dropoutTrg = options_->get<float>("dropout-trg");

      auto graph = embeddings->graph();

      if(dropoutTrg) {
        int trgWords = embeddings->shape()[2];
        auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
        embeddings = dropout(embeddings, mask=trgWordDrop);
      }

      auto context1 = context.first;
      auto context2 = context.second;

      auto contextMask1 = contextMask.first;
      auto contextMask2 = contextMask.second;

      if(!attention1_)
        attention1_ = New<GlobalAttention>("decoder",
                                          context1, dimDecState,
                                          mask=contextMask1,
                                          normalize=layerNorm);
      if(!attention2_)
        attention2_ = New<GlobalAttention>("decoder",
                                          context2, dimDecState,
                                          mask=contextMask2,
                                          normalize=layerNorm);

      RNN<MultiCGRU> rnnL1(graph, "decoder",
                      dimTrgEmb, dimDecState,
                      attention1_, attention2_,
                      normalize=layerNorm,
                      dropout_prob=dropoutRnn
                      );
      auto stateL1 = rnnL1(embeddings, states[0]);
      auto alignedContext1 = single ?
        rnnL1.getCell()->getLastContext1() :
        rnnL1.getCell()->getContexts1();

      auto alignedContext2 = single ?
        rnnL1.getCell()->getLastContext2() :
        rnnL1.getCell()->getContexts2();

      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < states.size(); ++i)
          statesIn.push_back(states[i]);

        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = MLRNN<GRU>(graph, "decoder",
                                                  decoderLayers - 1,
                                                  dimDecState, dimDecState,
                                                  normalize=layerNorm,
                                                  dropout_prob=dropoutRnn,
                                                  skip=skipDepth,
                                                  skip_first=skipDepth)
                                                 (stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (embeddings, outputLn, alignedContext1, alignedContext2);

      auto logitsL2 = Dense("ff_logit_l2", dimTrgVoc)
                        (logitsL1);

      return std::make_tuple(logitsL2, statesOut);
    }

};


template <class Encoder, class Decoder>
class MultiSeq2Seq {
  protected:
    Ptr<Config> options_;

    Ptr<EncoderBase> encoder1_;
    Ptr<EncoderBase> encoder2_;

    Ptr<DecoderBase> decoder_;

  public:

    MultiSeq2Seq(Ptr<Config> options)
     : options_(options),
       decoder_(New<Decoder>(options))
    {
       encoder1_ = New<Encoder>(options);
       encoder2_ = New<Encoder>(options);
    }

     virtual void load(Ptr<ExpressionGraph> graph,
                       const std::string& name) {
      graph->load(name);
    }

    virtual void save(Ptr<ExpressionGraph> graph,
                      const std::string& name) {
      graph->save(name);
    }

    virtual std::pair<std::tuple<std::vector<Expr>, Expr, Expr>>
    buildEncoder(Ptr<ExpressionGraph> graph,
                 Ptr<data::CorpusBatch> batch,
                 size_t encoderId) {

      using namespace keywords;
      graph->clear();

      encoder1_ = New<Encoder>(options);
      encoder2_ = New<Encoder>(options);

      decoder_ = New<Decoder>(options_);

      Expr srcContext1, srcMask1;
      std::tie(srcContext1, srcMask1) = encoder1_->build(graph, batch, 0);
      auto startState1 = decoder_->buildStartState(srcContext1, srcMask1);

      Expr srcContext2, srcMask2;
      std::tie(srcContext2, srcMask2) = encoder2_->build(graph, batch, 1);
      auto startState2 = decoder_->buildStartState(srcContext2, srcMask2);

      size_t decoderLayers = options_->get<size_t>("layers-dec");
      std::vector<Expr> startStates1(decoderLayers, startState1);
      std::vector<Expr> startStates2(decoderLayers, startState2);

      auto ret1 = std::make_tuple(startStates1, srcContext1, srcMask1);
      auto ret2 = std::make_tuple(startStates2, srcContext2, srcMask2);

      return std::make_pair(ret1, ret2);
    }

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         Expr context,
         Expr contextMask,
         bool single=false) {
      return decoder_->step(embeddings, states, context, contextMask, single);
    }

    virtual Expr build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      auto ret = buildEncoder(graph, batch);

      std::vector<Expr> startStates1;
      Expr srcContext1, srcMask1;
      std::tie(startStates1, srcContext1, srcMask1) = ret.first;

      std::vector<Expr> startStates2;
      Expr srcContext2, srcMask2;
      std::tie(startStates2, srcContext2, srcMask2) = ret.second;

      std::vector<Expr> startStates;
      for(int i = 0; i < startStates1; ++i) {
        startStates.push_back(startStates1[i] + startStates2[i]);
      }

      Expr trgEmbeddings, trgMask, trgIdx;
      std::tie(trgEmbeddings, trgMask, trgIdx) = decoder_->groundTruth(graph, batch, 2);

      Expr trgLogits;
      std::vector<Expr> trgStates;
      std::tie(trgLogits, trgStates) = decoder_->step(trgEmbeddings,
                                                      startStates,
                                                      std::make_pair(srcContext1, srcContext2),
                                                      std::make_pair(srcMask1, srcMask2));

      auto cost = CrossEntropyCost("cost")(trgLogits, trgIdx,
                                           mask=trgMask);

      return cost;
    }

};

}
