#include <set>
#include <string>
#include <boost/algorithm/string.hpp>

#include "training/config.h"
#include "common/file_stream.h"
#include "common/logging.h"

#define SET_OPTION(key, type) \
do { if(!vm_[key].defaulted() || !config_[key]) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)

#define SET_OPTION_NONDEFAULT(key, type) \
do { if(vm_.count(key) > 0) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)


namespace po = boost::program_options;

namespace marian {

size_t Config::seed = 1234;

bool Config::has(const std::string& key) const {
  return config_[key];
}

YAML::Node Config::get(const std::string& key) const {
  return config_[key];
}

const YAML::Node& Config::get() const {
  return config_;
}

YAML::Node& Config::get() {
  return config_;
}

void ProcessPaths(YAML::Node& node, const boost::filesystem::path& configPath, bool isPath) {
  using namespace boost::filesystem;
  std::set<std::string> paths = {"model", "trainsets", "vocabs"};

  if(isPath) {
    if(node.Type() == YAML::NodeType::Scalar) {
      std::string nodePath = node.as<std::string>();
      if (nodePath.size()) {
        try {
          node = canonical(path{nodePath}, configPath).string();
        } catch (boost::filesystem::filesystem_error& e) {
          std::cerr << e.what() << std::endl;
          auto parentPath = path{nodePath}.parent_path();
          node = (canonical(parentPath, configPath) / path{nodePath}.filename()).string();
        }
      }
    }

    if(node.Type() == YAML::NodeType::Sequence) {
      for (auto&& sub : node) {
        ProcessPaths(sub, configPath, true);
      }
    }
  }
  else {
    switch (node.Type()) {
      case YAML::NodeType::Sequence:
        for (auto&& sub : node) {
          ProcessPaths(sub, configPath, false);
        }
        break;
      case YAML::NodeType::Map:
        for (auto&& sub : node) {
          std::string key = sub.first.as<std::string>();
          ProcessPaths(sub.second, configPath, paths.count(key) > 0);
        }
        break;
    }
  }
}

void Config::validate(bool translate) const { 
    if(!translate) {
      UTIL_THROW_IF2(!has("train-sets")
                     || get<std::vector<std::string>>("train-sets").empty(),
                     "No train sets given in config file or on command line");
      if(has("vocabs")) {
        UTIL_THROW_IF2(get<std::vector<std::string>>("vocabs").size() !=
          get<std::vector<std::string>>("train-sets").size(),
          "There should be as many vocabularies as training sets");
      }
      if(has("valid-sets")) {
        UTIL_THROW_IF2(get<std::vector<std::string>>("valid-sets").size() !=
          get<std::vector<std::string>>("train-sets").size(),
          "There should be as many validation sets as training sets");
      }
    }
}

void Config::OutputRec(const YAML::Node node, YAML::Emitter& out) const {
  // std::set<std::string> flow = { "devices" };
  std::set<std::string> sorter;
  switch (node.Type()) {
    case YAML::NodeType::Null:
      out << node; break;
    case YAML::NodeType::Scalar:
      out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputRec(n, out);
      out << YAML::EndSeq;
      break;
    case YAML::NodeType::Map:
      for(auto& n : node)
        sorter.insert(n.first.as<std::string>());
      out << YAML::BeginMap;
      for(auto& key : sorter) {
        out << YAML::Key;
        out << key;
        out << YAML::Value;
        // if(flow.count(key))
          // out << YAML::Flow;
        OutputRec(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined:
      out << node; break;
  }
}

void Config::addOptionsCommon(po::options_description& desc) {
  po::options_description general("General options");
  general.add_options()
    ("config,c", po::value<std::string>(),
     "Configuration file")
    ("workspace,w", po::value<size_t>()->default_value(2048),
      "Preallocate  arg  MB of work space")
    ("log", po::value<std::string>(),
     "Log training process information to file given by  arg")
    ("seed", po::value<size_t>()->default_value(1234),
     "Seed for all random number generators")
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
      "Print this help message and exit")
  ;
  desc.add(general);
}

void Config::addOptionsModel(po::options_description& desc, bool translate=false) {
  po::options_description model("Model options");
  model.add_options()
    ("model,m", po::value<std::string>()->default_value("model.npz"),
      "Path prefix for model to be saved/resumed")
    ("type", po::value<std::string>()->default_value("dl4mt"),
      "Model type (possible values: dl4mt, gnmt, multi-gnmt")
    ("dim-vocabs", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({50000, 50000}), "50000 50000"),
      "Maximum items in vocabulary ordered by rank")
    ("dim-emb", po::value<int>()->default_value(512), "Size of embedding vector")
    ("dim-rnn", po::value<int>()->default_value(1024), "Size of rnn hidden state")
    ("layers-enc", po::value<int>()->default_value(1), "Number of encoder layers")
    ("layers-dec", po::value<int>()->default_value(1), "Number of decoder layers")
    ("skip", po::value<bool>()->zero_tokens()->default_value(false),
     "Use skip connections")
    ("layer-normalization", po::value<bool>()->zero_tokens()->default_value(false),
     "Enable layer normalization")
  ;

  if(!translate) {
    model.add_options()
      ("dropout-rnn", po::value<float>()->default_value(0),
       "Scaling dropout along rnn layers and time (0 = no dropout)")
      ("dropout-src", po::value<float>()->default_value(0),
       "Dropout source words (0 = no dropout)")
      ("dropout-trg", po::value<float>()->default_value(0),
       "Dropout target words (0 = no dropout)")
    ;
  }
  desc.add(model);
}

void Config::addOptionsTraining(po::options_description& desc) {
  po::options_description training("Training options");
  training.add_options()
    ("overwrite", po::value<bool>()->zero_tokens()->default_value(false),
      "Overwrite model with following checkpoints")
    ("no-reload", po::value<bool>()->zero_tokens()->default_value(false),
      "Do not load existing model specified in --model arg")
    ("train-sets,t", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to training corpora: source target")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --trainsets. "
      "If this parameter is not supplied we look for vocabulary files "
      "source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created.")
    ("max-length", po::value<size_t>()->default_value(50),
      "Maximum length of a sentence in a training sentence pair")
    ("after-epochs,e", po::value<size_t>()->default_value(0),
      "Finish after this many epochs, 0 is infinity")
    ("after-batches", po::value<size_t>()->default_value(0),
      "Finish after this many batch updates, 0 is infinity")
    ("disp-freq", po::value<size_t>()->default_value(1000),
      "Display information every  arg  updates")
    ("save-freq", po::value<size_t>()->default_value(10000),
      "Save model file every  arg  updates")
    ("no-shuffle", po::value<bool>()->zero_tokens()->default_value(false),
    "Skip shuffling of training data before each epoch")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for training. Asynchronous SGD is used with multiple devices.")
    ("mini-batch", po::value<int>()->default_value(64),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(100),
      "Number of batches to preload for length-based sorting")
    ("optimizer,o", po::value<std::string>()->default_value("adam"),
      "Optimization algorithm (possible values: sgd, adagrad, adam")
    ("learn-rate,l", po::value<double>()->default_value(0.0001),
      "Learning rate")
    ("clip-norm", po::value<double>()->default_value(1.f),
      "Clip gradient norm to  arg  (0 to disable)")
  ;
  desc.add(training);
}

void Config::addOptionsValid(po::options_description& desc) {
  po::options_description valid("Validation set options");
  valid.add_options()
    ("valid-sets", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to validation corpora: source target")
    ("valid-freq", po::value<size_t>()->default_value(10000),
      "Validate model every  arg  updates")
    ("valid-metrics", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"cross-entropy"}),
                      "cross-entropy"),
      "Metric to use during validation: cross-entropy, perplexity, valid-script. "
      "Multiple metrics can be specified")
    ("valid-script-path", po::value<std::string>(),
     "Path to external validation script")
    ("early-stopping", po::value<size_t>()->default_value(10),
     "Stop if the first validation metric does not improve for  arg  consecutive "
     "validation steps")
    ("valid-log", po::value<std::string>(),
     "Log validation scores to file given by  arg")
  ;
  desc.add(valid);
}

void Config::addOptionsTranslate(po::options_description& desc) {
  po::options_description translate("Translator options");
  translate.add_options()
    ("inputs,i", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to input files")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --inputs.")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for translating.")
    ("mini-batch", po::value<int>()->default_value(1),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(1),
      "Number of batches to preload for length-based sorting")
  ;
  desc.add(translate);
}

void Config::addOptions(int argc, char** argv,
                        bool doValidate, bool translate) {

  addOptionsCommon(cmdline_options_);

  addOptionsModel(cmdline_options_, translate);

  if(!translate) {
    addOptionsTraining(cmdline_options_);
    addOptionsValid(cmdline_options_);
  }
  else {
    addOptionsTranslate(cmdline_options_);
  }

  boost::program_options::variables_map vm_;
  try {
    po::store(po::command_line_parser(argc, argv)
              .options(cmdline_options_).run(), vm_);
    po::notify(vm_);
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;

    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options_ << std::endl;
    exit(1);
  }

  if (vm_["help"].as<bool>()) {
    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options_ << std::endl;
    exit(0);
  }

  std::string configPath;
  if(vm_.count("config")) {
    configPath = vm_["config"].as<std::string>();
    config_ = YAML::Load(InputFileStream(configPath));
  }
  else if(boost::filesystem::exists(vm_["model"].as<std::string>() + ".yml") &&
                                    !vm_["no-reload"].as<bool>()) {
    configPath = vm_["model"].as<std::string>() + ".yml";
    config_ = YAML::Load(InputFileStream(configPath));
  }

  /** model **/
  SET_OPTION("model", std::string);
  if (!vm_["vocabs"].empty()) {
    config_["vocabs"] = vm_["vocabs"].as<std::vector<std::string>>();
  }
  SET_OPTION("type", std::string);
  SET_OPTION("dim-vocabs", std::vector<int>);
  SET_OPTION("dim-emb", int);
  SET_OPTION("dim-rnn", int);
  SET_OPTION("layers-enc", int);
  SET_OPTION("layers-dec", int);
  SET_OPTION("skip", bool);
  SET_OPTION("layer-normalization", bool);
  if(!translate) {
    SET_OPTION("dropout-rnn", float);
    SET_OPTION("dropout-src", float);
    SET_OPTION("dropout-trg", float);
  }
  /** model **/

  /** training **/
  if(!translate) {
    SET_OPTION("overwrite", bool);
    SET_OPTION("no-reload", bool);
    if (!vm_["train-sets"].empty()) {
      config_["train-sets"] = vm_["train-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION("after-epochs", size_t);
    SET_OPTION("after-batches", size_t);
    SET_OPTION("disp-freq", size_t);
    SET_OPTION("save-freq", size_t);
    SET_OPTION("no-shuffle", bool);

    SET_OPTION("optimizer", std::string);
    SET_OPTION("learn-rate", double);
    SET_OPTION("clip-norm", double);
  }
  /** training **/
  else {
    if (!vm_["inputs"].empty()) {
      config_["inputs"] = vm_["inputs"].as<std::vector<std::string>>();
    }
  }

  /** valid **/
  if(!translate) {
    if (!vm_["valid-sets"].empty()) {
      config_["valid-sets"] = vm_["valid-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION_NONDEFAULT("valid-sets", std::vector<std::string>);
    SET_OPTION("valid-freq", size_t);
    SET_OPTION("valid-metrics", std::vector<std::string>);
    SET_OPTION_NONDEFAULT("valid-script-path", std::string);
    SET_OPTION("early-stopping", size_t);
    SET_OPTION_NONDEFAULT("valid-log", std::string);
  }
  /** valid **/

  if(doValidate) {
    try {
      validate(translate);
    }
    catch (util::Exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl;
  
      std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
      std::cerr << cmdline_options_ << std::endl;
      exit(1);
    }
  }
  
  SET_OPTION("workspace", size_t);
  SET_OPTION_NONDEFAULT("log", std::string);
  SET_OPTION("seed", size_t);
  SET_OPTION("relative-paths", bool);
  SET_OPTION("devices", std::vector<int>);
  SET_OPTION("mini-batch", int);
  SET_OPTION("maxi-batch", int);
  SET_OPTION("max-length", size_t);

  if (get<bool>("relative-paths") && !vm_["dump-config"].as<bool>())
    ProcessPaths(config_, boost::filesystem::path{configPath}.parent_path(), false);
  if(vm_["dump-config"].as<bool>()) {
    YAML::Emitter emit;
    OutputRec(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }
  seed = vm_["seed"].as<size_t>();
}

void Config::log() {
  createLoggers(*this);

  YAML::Emitter out;
  OutputRec(config_, out);
  std::string conf = out.c_str();

  std::vector<std::string> results;
  boost::algorithm::split(results, conf, boost::is_any_of("\n"));
  for(auto &r : results)
    LOG(config, r);
}

}
