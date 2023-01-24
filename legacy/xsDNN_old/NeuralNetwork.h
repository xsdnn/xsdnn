//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_NEURALNETWORK_H
#define XSDNN_NEURALNETWORK_H
// TODO: подумать, как собирать метаданные об обучении (ошибка, результаты по метрикам на тренировочном и валидационном
//  наборах)

# include <map>
# include <tuple>
# include "RNG.h"
# include "Utils/Random.h"
# include "Utils/InOut.h"
# include "Utils/CrLayer.h"
# include "Utils/variadic_table/include/VariadicTable.h"

namespace xsdnn {
    /*!
\brief Класс нейросети
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
    class NeuralNetwork {
    private:
        typedef std::map<std::string, Scalar> Meta;


        RNG m_default_rng;                  // дефолтный генератор
        RNG &m_rng;                          // генератор, преданный пользователем (ссылка на генератор), иначе дефолт

        std::vector<Layer *> m_layers;                       // указатели на созданные пользователем слои сетки
        Output *m_output;                       // указатель на выходной слой

        Scalar mean_loss;

        /// <summary>
        /// Проверка всех слоев на соотвествие вход текущего == выход предыдущего
        /// </summary>
        void check_unit_sizes() const {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 1) { return; }

            for (int i = 1; i < nlayer; ++i) {
                if (m_layers[i]->in_size() != m_layers[i - 1]->out_size()) {
                    throw internal::except::xs_error("[class NeuralNetwork]: Unit sizes do not match");
                }
            }
        }

        /// Инициализация необходимых параметров сети
        /// \param seed зерно генерации
        /// \param params параметры распределений
        void Init(int seed,
                  const std::vector<std::vector<Scalar>> &params) {
            check_unit_sizes();

            if (seed > 0) {
                m_rng.seed(seed);
            }

            const unsigned long nlayer = count_layers();

            if (params.empty()) {
                for (int i = 0; i < nlayer; i++) {
                    if (m_layers[i]->distribution_type() == "Uniform") {
                        const std::vector<Scalar> uniform_params = {0.0, 1.0};
                        m_layers[i]->init(uniform_params, m_rng);
                    }

                    if (m_layers[i]->distribution_type() == "Exponential") {
                        const std::vector<Scalar> exponential_params = {1.0};
                        m_layers[i]->init(exponential_params, m_rng);
                    }

                    if (m_layers[i]->distribution_type() == "Normal") {
                        const std::vector<Scalar> normal_params = {0.0, 1};
                        m_layers[i]->init(normal_params, m_rng);
                    }

                    if (m_layers[i]->distribution_type() == "Constant") {
                        const std::vector<Scalar> constant_params = {0.0};
                        m_layers[i]->init(constant_params, m_rng);
                    }
                }

                return;
            }

            if (params.size() != nlayer)
                throw internal::except::xs_error("[class NeuralNetwork] Distribution parameters vector size "
                                                 "does not match count layers. Check input data.");

            for (int i = 0; i < nlayer; ++i) {
                m_layers[i]->init(params[i], m_rng);
            }
        }

        /// <summary>
        /// Проход по всей сетке.
        /// </summary>
        /// <param name="input"> - входные данные.
        /// Убедитесь, что их длина равна длине входного слоя сетки</param>
        void Forward(const xsTypes::Matrix &input) {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return; }

            // Проверим нулевой слой на соотвествие правилу вход данных == вход нулевого слоя

            if (input.rows() != m_layers[0]->in_size()) {
                throw internal::except::xs_error("[class NeuralNetwork]: Input data have incorrect dimension");
            }

            // Протолкнули данные в нулевой слой

            m_layers[0]->forward(input);

            // Начинаем толкать данные по всей сетке

            for (unsigned long i = 1; i < nlayer; ++i) {
                m_layers[i]->forward(m_layers[i - 1]->output());
            }

            // На этом проход по всей сетке завершен
        }

        /// <summary>
        /// Backprop для всей сетки. Просто идем из конца сетки в ее начало и вызываем
        /// у каждого слоя метод обратного распространения
        /// </summary>
        /// <typeparam name="TargetType"> - тип таргета, отедльно для задач бинарной
        /// классификации и регрессии, а также для многоклассовой классификации</typeparam>
        /// <param name="input"> - входные данные</param>
        /// <param name="target"> - собственно таргет</param>
        void Backprop(const xsTypes::Matrix &input, const xsTypes::Matrix &target) {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return; }

            // Создадим указатель на первый и последний (скрытый, но не выходной)
            // слой сетки, это поможет в дальнейшем

            Layer *first_layer = m_layers[0];
            Layer *last_layer = m_layers[nlayer - 1];

            // Начнем распространение с конца сетки
            m_output->check_target_data(target);
            m_output->evaluate(last_layer->output(), target);

            // Если скрытый слой всего один, то 'prev_layer_data' будут выходными данными

            if (nlayer == 1) {
                first_layer->backprop(input,
                                      last_layer->backprop_data());
                return;
            }

            // Если это условие не выполнено, то вычисляем градиент для последнего скрытого слоя
            last_layer->backprop(m_layers[nlayer - 2]->output(),
                                 m_output->backprop_data());

            // Теперь пробегаемся по всем слоям и вычисляем градиенты

            for (unsigned long i = nlayer - 2; i > 0; --i) {
                m_layers[i]->backprop(m_layers[i - 1]->output(),
                                      m_layers[i + 1]->backprop_data());
            }

            // Теперь вычисляем грады для нулевого - входного слоя сетки

            first_layer->backprop(input,
                                  m_layers[1]->backprop_data());
        }

        /// <summary>
        /// Обновление весов модели
        /// </summary>
        /// <param name="opt"> - собственно оптимайзер</param>
        void Update(Optimizer &opt) {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return; }

            for (unsigned long i = 0; i < nlayer; ++i) {
                m_layers[i]->update(opt);
            }
        }

        void mean_loss_update() {
            mean_loss = mean_loss + this->get_output()->loss();
        }

        void mean_loss_reset() { mean_loss = 0.0; }


        /// <summary>
        /// Заполняем словарь для дальнейшего экспортирования сетки
        /// </summary>
        /// <returns></returns>
        Meta get_meta_info() const {
            const unsigned long nlayer = count_layers();
            Meta map;
            map.insert(std::make_pair("Nlayers", nlayer));

            // пробегаемся по всем слоям и вызываем метод сбора информации с одного слоя
            for (int i = 0; i < nlayer; ++i) {
                m_layers[i]->fill_meta_info(map, i);
            }

            // добавляем информацию о выходном слое
            map.insert(std::make_pair("OutputLayer", internal::output_id(m_output->output_type())));
            return map;
        }

    public:
        ///
        /// Стандартный конструктор
        ///

        NeuralNetwork() :
                m_default_rng(1),
                m_rng(m_default_rng),
                m_output(nullptr),
                mean_loss(0.0) {}

        ///
        /// Конструктор при передаче другого генератора
        ///

        explicit NeuralNetwork(RNG &rng) :
                m_default_rng(1),
                m_rng(rng),
                m_output(nullptr),
                mean_loss(0.0) {}

        ///
        /// Деструктор, удаляем из памяти все слои
        ///
        ~NeuralNetwork() {
            const unsigned long nlayer = count_layers();

            for (int i = 0; i < nlayer; ++i) {
                delete m_layers[i];
            }

            delete m_output;
        }

        /// <summary>
        /// Подсчет кол-ва слоев
        /// </summary>
        /// <returns></returns>
        unsigned long count_layers() const {
            return m_layers.size();
        }

        /// <summary>
        /// Добавить слой в сетку
        /// </summary>
        /// <param name="layer"> - указатель на слой</param>
        void add_layer(Layer *layer) {
            m_layers.push_back(layer);
        }

        void set_output(Output *output) {
            if (m_output) {
                delete m_output;
            }

            m_output = output;
        }

        /// <summary>
        /// None
        /// </summary>
        /// <returns>Получить выходной слой</returns>
        const Output *get_output() const {
            return m_output;
        }


        /// Установить рабочий процесс - тренировка
        void train() {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return; }

            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->train();
            }
        }

        /// Установить рабочий процесс - тестирование
        void eval() {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return; }

            for (int i = 0; i < nlayer; i++) {
                m_layers[i]->eval();
            }
        }

        // TODO: подумать что тут должно быть вместо xsTypes::Matrix
        bool fit(Optimizer &opt, const xsTypes::Matrix &x, const xsTypes::Matrix &y,
                 int batch_size, int epoch, int batch_seed = -1, int init_seed = -1,
                 const std::vector<std::vector<Scalar>> &params = std::vector<std::vector<Scalar>>()) {
            this->Init(init_seed, params);

            const unsigned int nsample = x.cols();             // кол-во объектов в выборке
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return false; }

            if (batch_size <= 0) {
                throw internal::except::xs_error("[class NeuralNetwork] Batch size must be greater than zero.");
            }

            // сбрасываем значения оптимизатора
            opt.reset();

            if (batch_seed > 0) {
                m_rng.seed(batch_seed);
            }

            // начинаем генерить батчи
            std::vector<xsTypes::Matrix> x_batches;
            std::vector<xsTypes::Matrix> y_batches;

            const int nbatch = internal::random::create_shuffled_batches(x, y, batch_size, m_rng, x_batches, y_batches);

#ifndef DNN_BE_QUIET
            internal::display::Timer t;
            internal::display::ProgressBar disp(nsample);
#endif

            this->train();

            // Начинаем процесс обучения
            for (int e = 0; e < epoch; ++e) {

                for (int i = 0; i < nbatch; ++i) {
                    this->Forward(x_batches[i]);
                    this->Backprop(x_batches[i], y_batches[i]);
                    this->Update(opt);
                    this->mean_loss_update();

                    // display update
#ifndef DNN_BE_QUIET
                    disp += batch_size;
#endif
                }

#ifndef DNN_BE_QUIET
                std::cout << "Epoch " << e + 1 << "/" << epoch << " completed. " << t.elapced() << "s elapsed."
                          << std::endl;
                std::cout << "Mean Loss per Epoch = " << mean_loss / nbatch << std::endl;
#endif
                this->mean_loss_reset();

#ifndef DNN_BE_QUIET
                if (e != epoch - 1) {
                    disp.restart(nsample);
                    t.restart();
                }
#endif
            }
            this->eval();
            return true;
        }

        // TODO: сделать через TensorBase
        xsTypes::Matrix predict(const xsTypes::Matrix &x) {
            const unsigned long nlayer = count_layers();

            if (nlayer <= 0) { return {}; }

            for (int i = 0; i < nlayer; i++) {
                if (m_layers[i]->get_workflow() != "eval") {
                    throw internal::except::xs_error(
                            "[class NeuralNetwork]: Model must be on eval workflow while predict. Set model process.");
                }
            }

            this->Forward(x);

            return m_layers[nlayer - 1]->output();
        }


        /// <summary>
        /// Получить параметры сетки
        /// </summary>
        /// <returns></returns>
        std::vector<std::vector<Scalar> > get_parameters() const {
            unsigned long nlayer = count_layers();
            std::vector<std::vector<Scalar> > res;
            res.reserve(nlayer); // зарезервировали место для большей оптимизации

            for (int i = 0; i < nlayer; ++i) {
                res.push_back(m_layers[i]->get_parametrs());
            }

            return res;
        }

        /// <summary>
        /// Установить пользовательские параметры сетки
        /// </summary>
        /// <param name="param"> - матрица параметров</param>
        void set_parameters(const std::vector<std::vector<Scalar> > &param) {
            unsigned long nlayer = count_layers();

            if (static_cast<int>(param.size()) != nlayer) {
                throw internal::except::xs_error("[class Neural Network]: param size does not match. check input param!");
            }

            for (int i = 0; i < nlayer; ++i) {
                m_layers[i]->set_parametrs(param[i]);
            }
        }


        /// <summary>
        /// Получить производные всех слоев
        /// </summary>
        /// <returns></returns>
        std::vector<std::vector<Scalar> > get_derivatives() const {
            unsigned long nlayer = count_layers();
            std::vector<std::vector<Scalar> > res;
            res.reserve(nlayer);

            for (int i = 0; i < nlayer; ++i) {
                res.push_back(m_layers[i]->get_derivatives());
            }

            return res;
        }

        /// Инициализация слоев согласно условиям
        /// \warning Этот метод следует использовать с осторожностью. Многие проверки выключены.
        /// \param seed зерно ГСЧ
        /// \param params вектор параметров распределений по слоям
        void init(int seed,
                  const std::vector<std::vector<Scalar>> &params) {
            this->Init(seed, params);
        }

        /// Генерация батчей
        /// \warning Этот метод следует использовать с осторожностью. Многие проверки выключены.
        /// \param x данные
        /// \param y метки
        /// \param batch_size размер батча
        /// \param batch_seed сид для генерации рандома
        /// \return кортеж из 2 векторов и кол-во батчей в данных
        auto generate_batch(const xsTypes::Matrix &x,
                            const xsTypes::Matrix &y,
                            const int batch_size,
                            const int batch_seed) -> std::tuple<std::vector<xsTypes::Matrix>, std::vector<xsTypes::Matrix>, int> {
            std::cout << "\x1B[31m[class NeuralNetwork] {auto generate_batch}"
                         "\n"
                         "You used advanced xsDNN_old method. Be careful...\033[0m\t\t";

            if (batch_size <= 0) {
                throw internal::except::xs_error("[class NeuralNetwork] Batch size must be greater than zero.");
            }

            if (batch_seed > 0) {
                m_rng.seed(batch_seed);
            }

            std::vector<xsTypes::Matrix> x_batches;
            std::vector<xsTypes::Matrix> y_batches;

            const int nbatch = internal::random::create_shuffled_batches(x, y, batch_size, m_rng, x_batches, y_batches);

            return {x_batches, y_batches, nbatch};
        }

        /// Проход вперед
        /// \warning Этот метод следует использовать с осторожностью. Многие проверки выключены.
        /// \param input входные данные (eq. train_data, test_data)
        void forward(const xsTypes::Matrix &input) {
            this->Forward(input);
        }

        /// Обратный проход
        /// \warning Этот метод следует использовать с осторожностью. Многие проверки выключены.
        /// \param input входные данные (eq. train_data, test_data)
        /// \param target метки классов / целевая переменная (eq. train_label, test_label)
        void backprop(const xsTypes::Matrix &input, const xsTypes::Matrix &target) {
            this->Backprop(input, target);
        }

        /// Обновление параметров сети - итерация градиентного спуска
        /// \warning Этот метод следует использовать с осторожностью. Многие проверки выключены.
        /// \param opt оптимизатор - объект класса Optimizer.
        void update(Optimizer &opt) {
            this->Update(opt);
        }

        friend NeuralNetwork &operator<<(NeuralNetwork &net, Layer *layer) {
            net.add_layer(layer);
            return net;
        }

        friend std::ostream &operator<<(std::ostream &output, NeuralNetwork &obj) {
            const unsigned long nlayer = obj.count_layers();
            VariadicTable<std::string, std::string, std::string, std::string> vt(
                    {"Layer", "Size", "Activation", "Distribution"}, 20);
            for (int i = 0; i < nlayer; i++) {
                if (obj[i]->layer_type() == "FullyConnected") {
                    vt.addRow("FullyConnected",
                              "(" + std::to_string(obj[i]->in_size()) + "," + std::to_string(obj[i]->out_size()) + ")",
                              obj[i]->activation_type(),
                              obj[i]->distribution_type());
                }

                if (obj[i]->layer_type() == "BatchNorm1D") {
                    vt.addRow("BatchNorm1D",
                              "(" + std::to_string(obj[i]->in_size()) + ")",
                              obj[i]->activation_type(),
                              obj[i]->distribution_type());
                }

                if (obj[i]->layer_type() == "Dropout") {
                    vt.addRow("Dropout",
                              "(" + std::to_string(obj[i]->in_size()) + ")",
                              obj[i]->activation_type(),
                              obj[i]->distribution_type());
                }
            }
            vt.print(output);
            return output;
        }

        /// Получить index-ый слой
        /// \param index
        /// \return
        Layer *operator[](const int &index) {
            if (index >= 0 || index < m_layers.size()) {
                return m_layers[index];
            }
            throw internal::except::xs_error("[class NeuralNetwork] Net index out of range.");
        }

        /// Сохранение сети
        /// \param folder название папки
        /// \param filename название файла модели. В одной папке может быть несколько моделей.
        void export_net(const std::string &folder, const std::string &filename) const {
            internal::io::create_directory(folder);

            std::vector<std::vector<Scalar>> params = this->get_parameters();
            Meta meta = this->get_meta_info();

            std::string directory_map = folder + "/" + filename;

            internal::io::write_map(directory_map, meta);
            internal::io::write_vector(folder, filename, params);

            std::cout << "NeuralNetwork saved" << std::endl;
        }


        /// Чтение и загрузка сети.
        /// \param folder папка с моделью
        /// \param filename название модели
        void read_net(const std::string &folder, const std::string &filename) {
            Meta map;
            std::string model_directory = folder + "/" + filename;

            internal::io::read_map(model_directory, map);

            int nlayer = map.find("Nlayers")->second;
            std::vector<std::vector<Scalar> > params = internal::io::read_parameter(folder, filename, nlayer);

            m_layers.clear();

            for (int i = 0; i < nlayer; i++) {
                m_layers.push_back(internal::create_layer(map, i));
            }

            this->set_parameters(params);
            this->set_output(internal::create_output(map));

            std::cout << "Net loaded successful" << std::endl;
        }
    };
} // namespace xsdnn



#endif //XSDNN_NEURALNETWORK_H
