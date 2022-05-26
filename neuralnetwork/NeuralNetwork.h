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


# include <iostream>
///
/// Этот модуль описывает интерфейс нейронной сети, которая будет использоваться пользователем
/// 

// TODO: доделать до конца сохранение и чтение сетки из файла

class NeuralNetwork
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::RowVectorXi IntegerVector;
	typedef std::map<std::string, int> Meta;


	RNG m_default_rng; // дефолтный генератор
	RNG& m_rng; // генератор, преданный пользователем (ссылка на генератор), иначе дефолт

	std::vector<Layer*> m_layers; // указатели на созданные пользователем слои сетки
	Output* m_output; // указатель на выходной слой
	Callback m_default_callback; // дефолтный вывод на печать
	Callback* m_callback; // пользовательский вывод на печать, иначе дефолт

	/// <summary>
	/// Проверка всех слоев на соотвествие вход текущего == выход предыдущего
	/// </summary>
	void check_unit_sizes() const
	{
		const int nlayer = count_layers();

		if (nlayer <= 1) { return; }

		for (int i = 1; i < nlayer; ++i)
		{
			if (m_layers[i]->in_size() != m_layers[i - 1]->out_size())
			{
				throw std::invalid_argument("[class NeuralNetwork]: Unit sizes do not match");
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
		const int nlayer = count_layers();

		if (nlayer <= 0) { return; }

		/// Проверим нулевой слой на соотвествие правилу вход данных == вход нулевого слоя

		if (input.rows() != m_layers[0]->in_size())
		{
			throw std::invalid_argument("[class NeuralNetwork]: Input data have incorrect dimension");
		}

		// Протолкнули данные в нулевой слой

		m_layers[0]->forward(input);

		// Начинаем толкать данные по всей сетке

		for (int i = 1; i < nlayer; ++i)
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
		const int nlayer = count_layers();

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

		for (int i = nlayer - 2; i > 0; --i)
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
		const int nlayer = count_layers();

		if (nlayer <= 0) { return; }

		for (int i = 0; i < nlayer; ++i)
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
		const int nlayer = count_layers();
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
		m_output(NULL),
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Конструктор при передаче другого генератора
	/// 

	NeuralNetwork(RNG& rng) :
		m_default_rng(1),
		m_rng(rng),
		m_output(NULL),
		m_default_callback(),
		m_callback(&m_default_callback)
	{}

	///
	/// Деструктор, удаляем из памяти все слои
	/// 
	~NeuralNetwork()
	{
		const int nlayer = count_layers();

		for (int i = 0; i < nlayer; ++i)
		{
			delete m_layers[i];
		}

		if (m_output)
		{
			delete m_output;
		}
	}

	/// <summary>
	/// Подсчет кол-ва слоев
	/// </summary>
	/// <returns></returns>
	int count_layers() const
	{
		return m_layers.size();
	}

	/// <summary>
	/// Получить последний скрытый слой для подсчета метрик классификации. 
	/// Используется в ClassificationCallback
	/// </summary>
	/// <returns></returns>
	const Matrix get_last_hidden_layer() const
	{
		const int nlayers = count_layers();
		return m_layers[nlayers - 1]->output();
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
	void init(const Scalar& mu = Scalar(0), const Scalar& sigma = Scalar(0.01), int seed = -1)
	{
		check_unit_sizes();

		if (seed > 0)
		{
			m_rng.seed(seed);
		}

		const int nlayer = count_layers();

		for (int i = 0; i < nlayer; ++i)
		{
			m_layers[i]->init(mu, sigma, m_rng);
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

		const int nlayer = count_layers();

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

		std::cout << "Batch init successfully!" << std::endl;

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
		const int nlayer = count_layers();

		if (nlayer <= 0) { return Matrix(); }

		this->forward(x);

		return m_layers[nlayer - 1]->output();
	}


	/// <summary>
	/// Получить параметры сетки
	/// </summary>
	/// <returns></returns>
	std::vector < std::vector<Scalar> > get_parameters() const
	{
		int nlayer = count_layers();
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
		int nlayer = count_layers();

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
		int nlayer = count_layers();
		std::vector < std::vector<Scalar> > res;
		res.reserve(nlayer);

		for (int i = 0; i < nlayer; ++i)
		{
			res.push_back(m_layers[i]->get_derivatives());
		}

		return res;
	}
};