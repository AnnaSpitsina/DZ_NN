#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

#include "back_propagation.hpp"
#include "neuron_layer.hpp"

namespace nn {
struct Network {
  std::vector<std::unique_ptr<NeuronLayer>> layers;  // вектор на указатель
                                                     // слоев

  Eigen::VectorXd predict(const Eigen::VectorXd& x) const {  // запуск нейронки
    if (layers.empty()) {
      return x;
    }

    Eigen::VectorXd x_i = x;  // вход для каждого последующего слоя
    for (size_t i = 0; i < layers.size(); ++i) {
      x_i = layers[i]->run(x_i);
    }

    return x_i;
  }
};
}  // namespace nn
