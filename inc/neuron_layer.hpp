#pragma once

#include <Eigen/Dense>
#include <functional>
#include <iostream>

#include "activate_func.hpp"

namespace nn {

class NeuronLayer {
 private:
  Eigen::MatrixXd weights_;  // веса, вектор размером количества нейронов на
                             // длину входного вектора
  std::shared_ptr<IActivateFunc>
      activate_func_;  // указатель на функцию активации
  Eigen::VectorXd x_;  // вектор результата функции активации
  Eigen::VectorXd z_;  // вектор результата производной функции активации
  Eigen::VectorXd error_;  // вектор ошибки в слое
  Eigen::VectorXd delta_;  // почленное произведение error_ и z_

 public:
  NeuronLayer(size_t in_size, size_t neuron_cnt,
              std::shared_ptr<IActivateFunc> func)
      : weights_(Eigen::MatrixXd::Random(neuron_cnt, in_size)),
        activate_func_(func),
        x_(neuron_cnt),
        z_(neuron_cnt),
        error_(neuron_cnt),
        delta_(neuron_cnt) {}
  ~NeuronLayer() {}

  // прогон через слой входных значений
  Eigen::VectorXd run(Eigen::VectorXd x) {
    return (weights_ * x)
        .unaryExpr(
            [this](double a) {  // входной вектор умножаем на матрицу весов
              return this->activate_func_->execute(
                  a);  // применяем для каждого значения получившегося вектора
                       // функцию активации
            });
  }

  // сеттеры и геттеры членов класса
  std::shared_ptr<IActivateFunc> get_activate_func() { return activate_func_; }
  Eigen::MatrixXd get_weights() const { return weights_; }
  Eigen::VectorXd get_x() const { return x_; }
  Eigen::VectorXd get_z() const { return z_; }
  Eigen::VectorXd get_error() const { return error_; }
  Eigen::VectorXd get_delta() const { return delta_; }

  void set_weights(Eigen::MatrixXd weights) { weights_ = weights; }
  void set_x(Eigen::VectorXd x) { x_ = x; }
  void set_z(Eigen::VectorXd z) { z_ = z; }
  void set_error(Eigen::VectorXd error) { error_ = error; }
  void set_delta(Eigen::VectorXd delta) { delta_ = delta; }
};
}  // namespace nn
