#pragma once

namespace nn {
class IActivateFunc {
 public:
  IActivateFunc() = default;
  virtual ~IActivateFunc() = default;

  virtual double execute(double) = 0;
  virtual double differentiate(double) = 0;
};

class ReLU : public IActivateFunc {
 public:
  ReLU() = default;
  ~ReLU() = default;

  double execute(double x) override { return x < 0 ? 0 : x; }
  double differentiate(double x) override { return x < 0 ? 0 : 1; }
};
}  // namespace nn
