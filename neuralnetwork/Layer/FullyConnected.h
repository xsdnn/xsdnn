#pragma once

# include <Eigen/Core>
# include <vector>
# include <stdexcept>
# include "../Config.h"
# include "../Layer.h"
# include "../Utils/Random.h"
# include "../Utils/Enum.h"


template <typename Activation>
class FullyConnected : public Layer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;
    typedef std::map<std::string, int> Meta;

    Matrix m_weight; // Веса модели
    Vector m_bias;   // Смещение весов
    Matrix m_dw;     // Производная весов
    Vector m_db;     // Производная смещения
    Matrix m_z;      // Значения нейронов до активации
    Matrix m_a;      // Значения нейронов после активации
    Matrix m_din;    // Проивзодная значений нейронов после backprop
    

public:
    FullyConnected(const int in_size, const int out_size, bool bias_true_false = true) :
        Layer(in_size, out_size, bias_true_false) {}

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
    {
        init();

        internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
        if (BIAS_ACTIVATE)  { internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma); }

        //std::cout << "Weights " << std::endl << m_weight << std::endl;
        //std::cout << "Bias " << std::endl << m_bias << std::endl;
    }

    void init()
    {
        m_weight.resize(this->m_in_size, this->m_out_size);
        m_dw.resize(this->m_in_size, this->m_out_size);
        if (BIAS_ACTIVATE)
        {
            m_bias.resize(this->m_out_size);
            m_db.resize(this->m_out_size);
        }
    }

    /// <summary>
    /// Описание того, как толкаются данные по сетке внутри одного слоя.
    /// Сначала получаем значения нейронов просто перемножая веса и предыдущие значения нейронов.
    /// Добавляем к этому смещение.
    /// Активируем.
    /// </summary>
    /// <param name="prev_layer_data"> - матрица значений нейронов предыдущего слоя</param>
    void forward(const Matrix& prev_layer_data)
    {
        const int ncols = prev_layer_data.cols();

        m_z.resize(this->m_out_size, ncols);
        m_z.noalias() = m_weight.transpose() * prev_layer_data;
        if (BIAS_ACTIVATE) { m_z.colwise() += m_bias; }
        m_a.resize(this->m_out_size, ncols);
        Activation::activate(m_z, m_a);
    }

    /// <summary>
    /// Возврат значений нейронов после активации
    /// </summary>
    /// <returns></returns>
    const Matrix& output() { return m_a; }

    /// <summary>
    /// Получаем производные этого слоя.
    /// 
    /// Нужно получить производные по 3 вещам.
    /// 
    /// 1. Производные весов - считаем Якобиан и умножаем на предыдущий слой
    /// 2. Производные смещения - среднее по строкам производных весов
    /// 3. Производные текущих значений нейронов - текущий вес на Якобиан.
    /// 
    /// Предыдущий / следующий слой считается слева направо.
    /// </summary>
    /// <param name="prev_layer_data"> - значения нейронов предыдущего слоя</param>
    /// <param name="next_layer_data"> - значения нейронов следующего слоя</param>
    void backprop(const Matrix& prev_layer_data,
        const Matrix& next_layer_data)
    {
        const int ncols = prev_layer_data.cols();

        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);
        m_dw.noalias() = prev_layer_data * dLz.transpose() / ncols;
        if (BIAS_ACTIVATE) { m_db.noalias() = dLz.rowwise().mean(); }
        m_din.resize(this->m_in_size, ncols);
        m_din.noalias() = m_weight * dLz;
    }

    /// <summary>
    /// Получить производную нейронов этого слоя
    /// </summary>
    /// <returns>ссылка на информацию</returns>
    const Matrix& backprop_data() const { return m_din; }


    /// <summary>
    /// Обновление весов и смещений используя переданный алгоритм оптимизации (см. Optimizer)
    /// </summary>
    /// <param name="opt"> - объект класса Optimizer</param>
    void update(Optimizer& opt)
    {
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        AlignedMapVec      w(m_weight.data(), m_weight.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec      b(m_bias.data(), m_bias.size());
    
        opt.update(dw, w);
        if (BIAS_ACTIVATE) { opt.update(db, b); }
        
    }

    /// <summary>
    /// Получить параметры одного слоя
    /// </summary>
    /// <returns>param - вектор параметров весов и смещения</returns>
    std::vector<Scalar> get_parametrs() const
    {
        if (BIAS_ACTIVATE)
        {
            std::vector<Scalar> res(m_weight.size() + m_bias.size()); // указали кол-во ячеек в этом векторе
            // просто копируем в этот массив все содержимое
            std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin()); // все аргументы передаются в виде указателей. Откуда начинаем, где заканчиваем, куда начинаем ложить.
            std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());
            return res;
        }
        else
        {
            std::vector<Scalar> res(m_weight.size());
            std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
            return res;
        }
        
    }

    /// <summary>
    /// Установить пользовательские параметры для одного слоя
    /// </summary>
    /// <param name="param"> - вектор значений параметров весов и смещений</param>
    void set_parametrs(const std::vector<Scalar>& param)
    {
        if (BIAS_ACTIVATE)
        {
            // сделаем проверку на равенство длин массивов
            // static_cast<int> - приведение длины массива к интовому типу данных
            if (static_cast<int>(param.size()) != (m_weight.size() + m_bias.size()))
            {
                throw std::invalid_argument("[class FullyConnected]: Parameter size does not match. Check parameter size!");
            }

            // если размеры сходятся, то копируем переданные значения в наши массивы
            std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
            std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data());
        }
        else
        {
            if (static_cast<int>(param.size()) != (m_weight.size()))
            {
                throw std::invalid_argument("[class FullyConnected]: Parameter size does not match. Check parameter size!");
            }

            // если размеры сходятся, то копируем переданные значения в наши массивы
            std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
        }
    }

    /// <summary>
    /// Получить производные весов и смщения одного слоя
    /// </summary>
    /// <returns></returns>
    std::vector<Scalar> get_derivatives() const
    {
        if (BIAS_ACTIVATE)
        {
            std::vector<Scalar> res(m_dw.size() + m_db.size());

            std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
            std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
            return res;
        }
        else
        {
            std::vector<Scalar> res(m_dw.size());
            std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
            return res;
        }

    }

    std::string layer_type() const { return "FullyConnected"; }

    std::string activation_type() const { return Activation::return_type(); }

    void fill_meta_info(Meta& map, int index) const
    {
        std::string ind = std::to_string(index);

        map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
        map.insert(std::make_pair("Activation " + ind, internal::activation_id(activation_type())));
        map.insert(std::make_pair("in_size " + ind, in_size()));
        map.insert(std::make_pair("out_size " + ind, out_size()));
    }
};