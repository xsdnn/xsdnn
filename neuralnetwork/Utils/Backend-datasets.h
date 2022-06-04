//
// Created by shuffle on 04.06.22.
//

#ifndef XSDNN_INCLUDE_BACKEND_DATASETS_H
#define XSDNN_INCLUDE_BACKEND_DATASETS_H

# include <string>
# include <utility>
# include <stdexcept>

# include "mnist/mnist_reader.hpp"
# include "mnist/mnist_reader_common.hpp"
# include "mnist/mnist_reader_less.hpp"
# include "mnist/mnist_utils.hpp"

namespace internal
{
    /// Класс отвечает за подгрузку данных и их обработку, заполняет структуру DataLoader.
    /// \tparam ImageFormat тип первоначальных данных для выборки - например, для изображения
    /// \tparam LabelFormat тип первоначальных данных для меток
    template <typename ImageFormat, typename LabelFormat>
    class DataFormat
    {
    private:
        std::string DATASET_NAME;

    public:
        explicit DataFormat(std::string dataset_name) : DATASET_NAME(std::move(dataset_name))
        {
            check_dataset_name();
        }

        void inline check_dataset_name()
        {
            if (DATASET_NAME != "mnist") throw std::invalid_argument("[class DataLoader] Dataset name is invalid.");
        }

        void load_data(Matrix& train_data, Matrix& train_label, Matrix& test_data, Matrix& test_label)
        {
            // TODO: Реализовать метод подгрузки данных
            mnist::MNIST_dataset<std::vector, std::vector<ImageFormat>, LabelFormat> dataset =
                    mnist::read_dataset<std::vector, std::vector, ImageFormat, LabelFormat>(MNIST_DATA_LOCATION);

            const long train_size_one_object = dataset.training_images[0].size();
            const long test_size_one_object = dataset.test_images[0].size();
            const long train_size = dataset.training_images.size();
            const long test_size = dataset.test_images.size();

            // resize matrix
            train_data.resize(train_size, train_size_one_object);
            train_label.resize(1, train_size);
            test_data.resize(test_size, test_size_one_object);
            test_label.resize(1, test_size);

            // TRAIN_DATA processing
            for (int i = 0; i < train_size; i++)
            {
                for (int j = 0; j < train_size_one_object; j++)
                {
                    train_data(i, j) = static_cast<Scalar>(dataset.training_images[i][j]);
                }
            }

            // TRAIN_LABEL processing
            for (int i = 0; i < train_size; i++)
            {
                train_label(0, i) = static_cast<Scalar>(dataset.training_labels[i]);
            }

            // TEST_DATA processing
            for (int i = 0; i < test_size; i++)
            {
                for (int j = 0; j < test_size_one_object; j++)
                {
                    test_data(i, j) = static_cast<Scalar>(dataset.test_images[i][j]);
                }
            }

            // TEST_LABEL processing
            for (int i = 0; i < test_size; i++)
            {
                test_label(0, i) = static_cast<Scalar>(dataset.test_labels[i]);
            }

            train_data.transposeInPlace();
            test_data.transposeInPlace();
        }
    };
}
/// Структура подгрузки данных для пользователя.
/// \tparam ImageFormat тип данных для обучения
/// \tparam LabelFormat тип меток
template <typename ImageFormat, typename LabelFormat>
struct DataLoader
{
public:
    Matrix train_data;
    Matrix train_label;
    Matrix test_data;
    Matrix test_label;

    explicit DataLoader(std::string data_name)
    {
        internal::DataFormat<ImageFormat, LabelFormat> obj = internal::DataFormat<ImageFormat, LabelFormat>(data_name);
        obj.load_data(train_data, train_label, test_data, test_label);
    }
};

#endif //XSDNN_INCLUDE_BACKEND_DATASETS_H
