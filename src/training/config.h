#pragma once

#include <boost/program_options.hpp>

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"

namespace marian {

class Config {
  public:
    
    static size_t seed;
    
    Config(int argc, char** argv, bool validate = true) {
      addOptions(argc, argv, validate);
      log();
    }

    bool has(const std::string& key) const;

    YAML::Node get(const std::string& key) const;

    template <typename T>
    T get(const std::string& key) const {
      return config_[key].as<T>();
    }

    const YAML::Node& get() const;
    YAML::Node& get();

    YAML::Node operator[](const std::string& key) const {
      return get(key);
    }

    void addOptions(int argc, char** argv, bool validate);
    void log();
    void validate() const;

    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      out << config.config_;
      return out;
    }

  private:
    std::string inputPath;
    YAML::Node config_;
};

}
