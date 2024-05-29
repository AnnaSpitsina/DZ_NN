#include <fstream>
#include <random>

#include "activate_func.hpp"
#include "back_propagation.hpp"

// гиперпараметры
constexpr size_t INPUT_VECTOR_SIZE = 1;
constexpr size_t OUTPUT_VECTOR_SIZE = 1;
constexpr size_t HIDDEN_LAYERS_NUM = 0;
constexpr size_t HIDDEN_LAYERS_SIZE = 0;
constexpr double LEARNING_RATE = 0.00005;

constexpr size_t EPOCH_CNT = 3;  // сколько раз обучаемся на данных для обучения
constexpr size_t TRAIN_VECTOR_SIZE = 10;  // количество данных для обучения
constexpr size_t TEST_VECTOR_SIZE = 5;  // количество данных для проверки

constexpr size_t NOISE_MAX = 3;

double target_func(double x) { return 5.5 * x; }

// создаем нейронную сеть
nn::Network build_nn() {
  nn::Network net;
  std::shared_ptr<nn::ReLU> activate_func = std::make_shared<nn::ReLU>();

  net.layers.push_back(std::make_unique<nn::NeuronLayer>(
      INPUT_VECTOR_SIZE,
      HIDDEN_LAYERS_NUM ? HIDDEN_LAYERS_SIZE : OUTPUT_VECTOR_SIZE,
      activate_func));

  for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i) {
    net.layers.push_back(std::make_unique<nn::NeuronLayer>(
        HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE, activate_func));
  }

  if (HIDDEN_LAYERS_NUM) {
    net.layers.push_back(std::make_unique<nn::NeuronLayer>(
        HIDDEN_LAYERS_SIZE, OUTPUT_VECTOR_SIZE, activate_func));
  }
  return net;
}

// обучение
void train(nn::Network& net) {
  Eigen::MatrixXd train_x(1, TRAIN_VECTOR_SIZE);
  Eigen::MatrixXd train_y(1, TRAIN_VECTOR_SIZE);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis_noise(0, NOISE_MAX);  // нормальное распред-е
  std::uniform_real_distribution<> dis_x(
      0, TRAIN_VECTOR_SIZE * 10);  // равномерное распред-е

  for (int i = 0; i < TRAIN_VECTOR_SIZE; ++i) {
    double x = dis_x(gen);  // берем разбросанные значения х для обучения
    double y = target_func(x) + dis_noise(gen);  // добавляем шум
    train_x(0, i) = x;
    train_y(0, i) = y;
  }

  nn::BackPropagation bp(LEARNING_RATE);
  for (size_t i = 0; i < EPOCH_CNT; ++i) {
    for (size_t j = 0; j < TRAIN_VECTOR_SIZE; ++j) {
      bp.move(net, train_x.col(j), train_y.col(j));
    }
  }
}

// проверка средней ошибки
void check(const nn::Network& net) {
  Eigen::VectorXd total_error({{0}});
  for (double i = TRAIN_VECTOR_SIZE; i < TRAIN_VECTOR_SIZE + TEST_VECTOR_SIZE;
       ++i) {
    Eigen::VectorXd x({{i}});
    Eigen::VectorXd y = 5.5 * x;
    Eigen::VectorXd prediction = net.predict(x);
    total_error += y - prediction;
  }
  for (int i = 0; i < net.layers.size(); ++i) {
    std::cout << "W: \n" << net.layers[i]->get_weights() << std::endl;
  }
  std::cout << "ERROR: " << total_error / TEST_VECTOR_SIZE << std::endl;
}

// сохранение в файл для картинки
void save_file(const nn::Network& net) {
  std::ofstream outfile("results.csv");
  outfile << "x,ideal,noise,nn_output" << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis_noise(0, NOISE_MAX);

  for (double x = 0; x <= 10; x += 0.1) {
    double ideal_y = target_func(x);
    double noise_y = ideal_y + dis_noise(gen);
    Eigen::VectorXd input({{x}});
    double nn_output = net.predict(input)(0);
    outfile << x << "," << ideal_y << "," << noise_y << "," << nn_output
            << std::endl;
  }

  outfile.close();
}

int main() {
  nn::Network net = build_nn();

  train(net);

  check(net);

  // save_file(net);

  return 0;
}
