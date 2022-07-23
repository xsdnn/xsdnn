#pragma once

# include <Eigen/Core>
# include <map>
# include <vector>
# include <stdexcept>
# include "Config.h"
# include "RNG.h"
# include "Utils/Random.h"
# include "Layer.h"
# include "Output.h"
# include "Callback.h"
# include "Utils/Enum.h"
# include "Utils/InOut.h"
# include "Utils/CrLayer.h"

# include <iostream>
# include <iomanip>
///
/// Этот модуль описывает интерфейс нейронной сети, которая будет использоваться пользователем
/// 

// TODO: подумать, требуется ли в этой библиотеки метод перевода сети в тренировочный и отладочный режимы?...
// Эти режимы нужны для того, чтобы понять когда переключать некоторые слои в режимы обучения и тестирования, например,
// это могут быть слои: Dropout, BatchNorm...
//
// Если получится, что по другому эту возможность реализовать нельзя, то определить как будет реализована эта возможность.

class NeuralNetwork
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef std::map<std::string, int> Meta;


	RNG m_default_rng;                  // дефолтный генератор
	RNG& m_rng;                         // генератор, преданный пользователем (ссылка на генератор), иначе дефолт

	std::vector<Layer*> m_layers;       // указатели на созданные пользователем слои сетки
	Output* m_output;                   // указатель на выходной слой
	Callback m_default_callback;        // дефолтный вывод на печать
	Callback* m_callback;               // пользовательский вывод на печать, иначе дефолт

	/// <summary>
	/// Проверка всех слоев на соотвествие вход текущего == выход предыдущего
	/// </summary>
	void check_unit_sizes() const
	{
		const unsigned long nlayer = count_layers();

		if (nlayer <= 1) { return; }

		for (int i = 1; i < nlayer; ++i)
		{
			if (m_layers[i]->in_size() != m_layers[i - 1]->out_size())
			{
				throw std::invalid_argument("[class NeuralNetwork]: Unit sizes do not match");
			}
		}
	}

    void check_unit_workflow() const
    {
        const unsigned long nlayer = count_layers();

        if (nlayer <= 0) { return; }

        for (int i = 0; i < nlayer; i++)
        {
            if (m_layers[i]->get_workflow() == "undefined")
            {
                throw std::invalid_argument("[class NeuralNetwork]: Model must be on train or eval workflow. Set model process.");
            }
        }
    }

	/// <summary>
	/// Проход по всей сетке.
	/// </summary>
	/// <param name="input"> - входные данные. 
	/// Убедитесь, что их длина равна длине входного слоя сетки</param>
	void forward(const Matrix& input)
	{
		const unsigned long nlayer = count_layers();

		if (nlayer <= 0) { return; }

		/// Проверим нулевой слой на соотвествие правилу вход данных == вход нулевого слоя

		if (input.rows() != m_layers[0]->in_size())
		{
			throw std::invalid_argument("[class NeuralNetwork]: Input data have incorrect dimension");
		}

		// Протолкнули данные в нулевой слой

		m_layers[0]->forward(input);

		// Начинаем толкать данные по всей сетке

		for (unsigned long i = 1; i < nlayer; ++i)
		{
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
	template <typename TargetType>
	void backprop(const Matrix& input, const TargetType& target)
	{
		const unsigned long nlayer = count_layers();

		if (nlayer <= 0) { return; }

		// Создадим указатель на первый и последний (скрытый, но не выходной)
		// слой сетки, это поможет в дальнейшем

		Layer* first_layer = m_layers[0];
		Layer* last_layer = m_layers[nlayer - 1];

		// Начнем распространение с конца сетки
		m_output->check_target_data(target);
		m_output->evaluate(last_layer->output(), target);

		// Если скрытый слой всего один, то 'prev_layer_data' будут выходными данными

		if (nlayer == 1)
		{
			first_layer->backprop(input, last_layer->backprop_data());
			return;
		}

		// Если это условие не выполнено, то вычисляем градиент для последнего скрытого слоя
		last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());

		// Теперь пробегаемся по всем слоям и вычисляем градиенты

		for (unsigned long i = nlayer - 2; i > 0; --i)
		{
			m_layers[i]->backprop(m_layers[i - 1]->output(),
				m_layers[i + 1]->backprop_data());
		}

		// Теперь вычисляем грады для нулевого - входного слоя сетки

		first_layer->backprop(input, m_layers[1]->backprop_data());

		// На этом backprop окончен
	}

	/// <summary>
	/// Обновление весов модели
	/// </summary>
	/// <param name="opt"> - собственно оптимайзер</param>
	void update(Optimizer& opt)
	{
		const unsigned long nlayer = count_layers();

		if (nlayer <= 0) { return; }

		for (unsigned long i = 0; i < nlayer; ++i)
		{
			m_layers[i]->update(opt);
		}
	}

	/// <summary>
	/// Заполняем словарь для дальнейшего экспортирования сетки
	/// </summary>
	/// <returns></returns>
	Meta get_meta_info() const
	{
		const unsigned long nlayer = count_layers();
		Meta map;
		map.insert(std::make_pair("Nlayers", nlayer));

		// пробегаемся по всем слоям и вызываем метод сбора информации с одного слоя
		for (int i = 0; i < nlayer; ++i)
		{
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
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Конструктор при передаче другого генератора
	/// 

	explicit NeuralNetwork(RNG& rng) :
		m_default_rng(1),
		m_rng(rng),
		m_output(nullptr),
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Деструктор, удаляем из памяти все слои
	/// 
	~NeuralNetwork()
	{
		const unsigned long nlayer = count_layers();

		for (int i = 0; i < nlayer; ++i)
		{
			delete m_layers[i];
		}

        delete m_output;
	}

	/// <summary>
	/// Подсчет кол-ва слоев
	/// </summary>
	/// <returns></returns>
	unsigned long count_layers() const
	{
		return m_layers.size();
	}

	/// <summary>
	/// Добавить слой в сетку
	/// </summary>
	/// <param name="layer"> - указатель на слой</param>
	void add_layer(Layer* layer)
	{
		m_layers.push_back(layer);
	}

	void set_output(Output* output)
	{
		if (m_output)
		{
			delete m_output;
		}

		m_output = output;
	}

	/// <summary>
	/// None
	/// </summary>
	/// <returns>Получить выходной слой</returns>
	const Output* get_output() const
	{
		return m_output;
	}

	/// <summary>
	/// Установить пользовательский вывод информации про обучение сетки.
	/// </summary>
	/// <param name="callback"> - ссылка на объект класса, который будет здесь работать.</param>
	void set_callback(Callback& callback)
	{
		m_callback = &callback;
	}

	/// <summary>
	/// Установить дефолтный вывод.
	/// </summary>
	void set_default_callback()
	{
		m_callback = &m_default_callback;
	}


	/// <summary>
	/// Инициализация слоев сетки. Первая генерация весов сетки 
	/// -
	///  значения из нормального распределения.
	/// </summary>
	/// <param name="mu"> - мат. ожидание нормального распределения.</param>
	/// <param name="sigma"> - дисперсия нормального распределения.</param>
	/// <param name="seed"> - сид для рандома.</param>
	void init(int seed = -1, const std::vector<std::vector<Scalar>>& params = std::vector<std::vector<Scalar>>())
	{
		check_unit_sizes();

		if (seed > 0)
		{
			m_rng.seed(seed);
		}

		const unsigned long nlayer = count_layers();

        if (params.empty())
        {
            for (int i = 0; i < nlayer; i++)
            {
                if (m_layers[i]->distribution_type() == "Uniform")
                {
                    const std::vector<Scalar> uniform_params = {0.0, 1.0};
                    m_layers[i]->init(uniform_params, m_rng);
                }

                if (m_layers[i]->distribution_type() == "Exponential")
                {
                    const std::vector<Scalar> exponential_params = {1.0};
                    m_layers[i]->init(exponential_params, m_rng);
                }

                if (m_layers[i]->distribution_type() == "Normal")
                {
                    const std::vector<Scalar> normal_params = {0.0, 1};
                    m_layers[i]->init(normal_params, m_rng);
                }

                // TODO: add some other distribution
            }

            return;
        }

        if (params.size() != nlayer) throw std::length_error("[class NeuralNetwork] Distribution parameters vector size "
                                                             "does not match count layers. Check input data.");

		for (int i = 0; i < nlayer; ++i)
		{
			m_layers[i]->init(params[i], m_rng);
		}
	}

    /// Установить рабочий процесс - тренировка
    void train()
    {
        const unsigned long nlayer = count_layers();

        if (nlayer <= 0) { return; }

        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->train();
        }
    }

    /// Установить рабочий процесс - тестирование
    void eval()
    {
        const unsigned long nlayer = count_layers();

        if (nlayer <= 0) { return; }

        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->eval();
        }
    }

	/// <summary>
	/// Начать обучение сетки
	/// </summary>
	/// <typeparam name="DerivedX"></typeparam>
	/// <typeparam name="DerivedY"></typeparam>
	/// <param name="opt"> - оптимизатор</param>
	/// <param name="x"> - вектор для обучения</param>
	/// <param name="y"> - таргет</param>
	/// <param name="batch_size"> - размер батча</param>
	/// <param name="epoch"> - кол-во эпох при обучении сетки</param>
	/// <param name="seed"> - сид для генерации случайных чисел</param>
	/// <returns>True если все прошло хорошо</returns>
	template <typename DerivedX, typename DerivedY>
	bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x,
		const Eigen::MatrixBase<DerivedY>& y,
		int batch_size, int epoch, int seed = -1)
	{
		typedef typename Eigen::MatrixBase<DerivedX>::PlainObject PlainObjectX;
		typedef typename Eigen::MatrixBase<DerivedY>::PlainObject PlainObjectY;
		typedef Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime>
			XType;
		typedef Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime>
			YType;

        check_unit_workflow();

		const unsigned long nlayer = count_layers();

		if (nlayer <= 0) { return false; }

		// сбрасываем значения оптимизатора
		opt.reset();

		if (seed > 0)
		{
			m_rng.seed(seed);
		}

		// начинаем генерить батчи
		std::vector<XType> x_batches;
		std::vector<YType> y_batches;

		const int nbatch = internal::create_shuffled_batches(x, y, batch_size, m_rng, x_batches, y_batches);

		// Передаем параметры в callback для дальнейшего отслеживания обучения

		m_callback->m_nbatch = nbatch;
		m_callback->m_nepoch = epoch;

		// Начинаем процесс обучения
		for (int e = 0; e < epoch; ++e)
		{
			m_callback->m_epoch_id = e;

			for (int i = 0; i < nbatch; ++i)
			{
				m_callback->m_batch_id = i;
				m_callback->pre_trained_batch(this, x_batches[i], y_batches[i]);
				this->forward(x_batches[i]);
				this->backprop(x_batches[i], y_batches[i]);
				this->update(opt);
				m_callback->post_trained_batch(this, x_batches[i], y_batches[i]);
			}
		}

		return true;
	}

	Matrix predict(const Matrix& x)
	{
		const unsigned long nlayer = count_layers();

		if (nlayer <= 0) { return {}; }

        for (int i = 0; i < nlayer; i++)
        {
            if (m_layers[i]->get_workflow() != "eval")
            {
                throw std::invalid_argument("[class NeuralNetwork]: Model must be on eval workflow while predict. Set model process.");
            }
        }

		this->forward(x);

		return m_layers[nlayer - 1]->output();
	}


	/// <summary>
	/// Получить параметры сетки
	/// </summary>
	/// <returns></returns>
	std::vector < std::vector<Scalar> > get_parameters() const
	{
		unsigned long nlayer = count_layers();
		std::vector < std::vector<Scalar> > res;
		res.reserve(nlayer); // зарезервировали место для большей оптимизации

		for (int i = 0; i < nlayer; ++i)
		{
			res.push_back(m_layers[i]->get_parametrs());
		}

		return res;
	}

	/// <summary>
	/// Установить пользовательские параметры сетки
	/// </summary>
	/// <param name="param"> - матрица параметров</param>
	void set_parameters(const std::vector < std::vector<Scalar> >& param)
	{
		unsigned long nlayer = count_layers();

		if (static_cast<int>(param.size()) != nlayer)
		{
			throw std::invalid_argument("[class Neural Network]: param size does not match. check input param!");
		}

		for (int i = 0; i < nlayer; ++i)
		{
			m_layers[i]->set_parametrs(param[i]);
		}
	}


	/// <summary>
	/// Получить производные всех слоев
	/// </summary>
	/// <returns></returns>
	std::vector < std::vector<Scalar> > get_derivatives() const
	{
		unsigned long nlayer = count_layers();
		std::vector < std::vector<Scalar> > res;
		res.reserve(nlayer);

		for (int i = 0; i < nlayer; ++i)
		{
			res.push_back(m_layers[i]->get_derivatives());
		}

		return res;
	}

    friend std::ostream& operator << (std::ostream& output, NeuralNetwork& obj)
    {
        output << "Neural Network consists of elements" << std::endl;
        output << std::setw(14) << "" << "Layer" << std::setw(8) << "" << "Activation"
                << std::setw(8) << "" << "Distribution" <<std::endl;

        const unsigned long nlayer = obj.count_layers();
        if (nlayer == 0) return output;
        for (unsigned long i = 0; i < nlayer; i++)
        {
            output << std::setw(10) << "" << obj.m_layers[i]->layer_type()
            << std::setw(5) << "" << obj.m_layers[i]->activation_type()
            << std::setw(11) << "" << obj.m_layers[i]->distribution_type()
            << std::setw(7) << "" << "Input neuron = " << obj.m_layers[i]->in_size()
            << std::setw(7) << "" << "Output neuron = " << obj.m_layers[i]->out_size() << std::endl;
        }
        return output;
    }

    /// Сохранение сети
    /// \param folder название папки
    /// \param filename название файла модели. В одной папке может быть несколько моделей.
    void export_net(const std::string& folder, const std::string& filename) const
    {
        internal::create_directory(folder);

        std::vector <std::vector<Scalar>> params = this->get_parameters();
        Meta meta = this->get_meta_info();

        std::string directory_map = "../xsDNN-models/" + folder + "/" + filename;
        std::string directory_vector = "../xsDNN-models/" + folder;

        internal::write_map(directory_map, meta);
        internal::write_vector(directory_vector, filename, params);

        std::cout << "NeuralNetwork saved" << std::endl;
    }


    /// Чтение и загрузка сети.
    /// \param folder папка с моделью
    /// \param filename название модели
    void read_net(const std::string& folder, const std::string& filename)
    {
        Meta map;
        std::string model_directory = folder + "/" + filename;

        internal::read_map(model_directory, map);

        int nlayer = map.find("Nlayers")->second;
        std::vector< std::vector<Scalar> > params = internal::read_parameter(folder, filename, nlayer);

        m_layers.clear();

        for (int i = 0; i < nlayer; i++)
        {
            m_layers.push_back(internal::create_layer(map, i));
        }

        this->set_parameters(params);
        this->set_output(internal::create_output(map));

        std::cout << "Net loaded successful" << std::endl;
    }
};