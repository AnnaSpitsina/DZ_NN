#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>

#include "network.hpp"

namespace nn {
class BackPropagation {
 private:
  double learning_rate_;  // Скорость обучения

  // прямой проход МОРО
  void forward(Network& network, Eigen::VectorXd train_x) {
    Eigen::VectorXd x_i = train_x;

    for (int i = 0; i < network.layers.size(); ++i) {
      Eigen::VectorXd v = network.layers[i]->get_weights() * x_i;
      network.layers[i]->set_x(
          v.unaryExpr([func = network.layers[i]->get_activate_func()](
                          double a) { return func->execute(a); }));
      network.layers[i]->set_z(
          v.unaryExpr([func = network.layers[i]->get_activate_func()](
                          double a) { return func->differentiate(a); }));
      x_i = network.layers[i]->get_x();
    }
  }

  // обратный проход МОРО
  void backward(Network& network, Eigen::VectorXd train_y) {
    network.layers.back()->set_error(network.layers.back()->get_x() - train_y);
    network.layers.back()->set_delta(
        (network.layers.back()->get_error().cwiseProduct(
            network.layers.back()->get_z())));

    for (int i = network.layers.size() - 2; i >= 0; --i) {
      network.layers[i]->set_error(
          network.layers[i + 1]->get_weights().transpose() *
          network.layers[i + 1]->get_delta());
      network.layers[i]->set_delta(network.layers[i]->get_error().cwiseProduct(
          network.layers[i]->get_z()));
    }
  }

  // обновление весов
  void update(Network& network, Eigen::VectorXd train_x) {
    auto x_prev = train_x;
    for (int i = 0; i < network.layers.size(); ++i) {
      network.layers[i]->set_weights(
          network.layers[i]->get_weights() -
          learning_rate_ *
              Eigen::kroneckerProduct(network.layers[i]->get_delta(),
                                      x_prev.transpose()));
      x_prev = network.layers[i]->get_x();
    }
  }

 public:
  BackPropagation(double lr) : learning_rate_(lr) {}
  ~BackPropagation() {}

  // шаг обучения
  void move(Network& network, Eigen::VectorXd train_x,
            Eigen::VectorXd train_y) {
    forward(network, train_x);
    backward(network, train_y);
    update(network, train_x);
  }
};
}  // namespace nn
