# xsDNN

Библиотека в разработке. Документация доступна по __[ссылке](https://shuffle-true.github.io/xsDNN-page/)__

# Установка

```bash
git clone https://github.com/shuffle-true/xsDNN.git && cd xsDNN_old && git submodule init && git submodule update && bash ./install_old.sh && source ~/.bashrc
```

#### Обучим простую сеть на MNISTе.


# Функция обучения

## Загрузка данных

```cpp
Matrix train_image, train_label;

dataset::parse_mnist_image("../datasets/mnist/train-images-idx3-ubyte",
                           train_image,
                           0,
                           1,
                           0,
                           0);

dataset::parse_mnist_label("../datasets/mnist/train-labels-idx1-ubyte", train_label);
```

## Построение архитектуры сети

```cpp
NeuralNetwork baseline;

baseline    << new FullyConnected<init::Normal, activate::ReLU>(784, 128)
            << new FullyConnected<init::Normal, activate::Softmax>(128, 10);

Output* criterion = new CrossEntropyLoss();
baseline.set_output(criterion);

std::vector< std::vector<Scalar> > init_params = {
        {0.0, 1.0 / (784.0 + 128.0)},
        {0.0, 1.0 / (128.0 + 10.0)}
};

SGD opt; opt.m_lrate = 0.01;
```

## Обучение и сохранение

```cpp
baseline.fit(opt, train_image, train_label, 16, 5, 42);
baseline.export_net("example_old-mnist", "baseline");
```

 <br>
 <br>
 <br>

# Функция тестирования

## Загрузка данных

```cpp
Matrix test_image, test_label;
dataset::parse_mnist_image("../datasets/mnist/t10k-images-idx3-ubyte",
                            test_image,
                            0.0,
                            1.0,
                            0.0,
                            0.0);
dataset::parse_mnist_label("../datasets/mnist/t10k-labels-idx1-ubyte", test_label);
```

## Чтение сети и предсказание


```cpp
NeuralNetwork net;
net.read_net("example_old-mnist", "baseline");

net.eval();
Matrix predict = net.predict(test_image);
```


## Подсчет метрики

```cpp
void calculate_accuracy(const Matrix& target, const Matrix& predict) {
    const int nobj = target.cols();
    auto argmax_vec = internal::math::colargmax_vector(predict);
    Scalar nom = 0;
    for (int i = 0; i < nobj; i++) {
        for (int j = 0; j < 10; j++) {
            if (target(j, i) == Scalar(1)) {
                if (j == argmax_vec[i]) nom++;
            }
        }
    }
    std::cout << "Accuracy = " << nom / nobj << "%" << std::endl;
}
```

Полный исходный код доступен __[здесь](https://github.com/shuffle-true/dl_new/tree/main/example/mnist)__.
