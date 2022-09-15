//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_INCLUDE_MNIST_IO_H
#define XSDNN_INCLUDE_MNIST_IO_H

namespace internal {
    template<typename T>
    T *reverse_endian(T *p) {
        std::reverse(reinterpret_cast<char *>(p),
                     reinterpret_cast<char *>(p) + sizeof(T));
        return p;
    }

    inline bool is_little_endian() {
        int x = 1;
        return *reinterpret_cast<char *>(&x) != 0;
    }

    struct mnist_header {
        uint32_t magic_number;
        uint32_t num_items;
        uint32_t num_rows;
        uint32_t num_cols;
    };

    inline void parse_mnist_header(std::ifstream &ifs, mnist_header &header) {
        ifs.read(reinterpret_cast<char *>(&header.magic_number), 4);
        ifs.read(reinterpret_cast<char *>(&header.num_items), 4);
        ifs.read(reinterpret_cast<char *>(&header.num_rows), 4);
        ifs.read(reinterpret_cast<char *>(&header.num_cols), 4);

        if (is_little_endian()) {
            reverse_endian(&header.magic_number);
            reverse_endian(&header.num_items);
            reverse_endian(&header.num_rows);
            reverse_endian(&header.num_cols);
        }

        if (header.magic_number != 0x00000803 || header.num_items <= 0)
            throw internal::except::xs_error("[inline void parse_mnist_label] MNIST label-file format error");
        if (ifs.fail() || ifs.bad()) throw internal::except::xs_error("[inline void parse_mnist_label] file error");
    }

    inline void parse_mnist_image(std::ifstream &ifs,
                                  const mnist_header &header,
                                  const int width,
                                  const int height,
                                  const Scalar &scale_min,
                                  const Scalar &scale_max,
                                  const int x_padding,
                                  const int y_padding,
                                  Eigen::VectorXd &col) {
        typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> Vector;

        const int w = header.num_cols + 2 * x_padding;
        const int h = header.num_rows + 2 * y_padding;

        Vector image_vec(header.num_rows * header.num_cols);

        ifs.read(reinterpret_cast<char *>(image_vec.data()), header.num_rows * header.num_cols);

        col.resize(w * h);
        Scalar *arr = col.data();
        unsigned char *image_vec_arr = image_vec.data();

        for (uint32_t y = 0; y < header.num_rows; y++)
            for (uint32_t x = 0; x < header.num_cols; x++) {
                arr[w * (y + y_padding) + x + x_padding] =
                        (image_vec_arr[y * header.num_cols + x] / Scalar(255)) *
                        (scale_max - scale_min) +
                        scale_min;
            }
    }
}
/// \details Подгрузка дата-сетов
namespace dataset {
    /// \brief Чтение меток дата-сета MNIST
    /// \warning параметр image_filename должен содержать полный путь до файла
    /// \param label_filename расположение файла метод (i.e. ../some/path/to/train-images-idx1-ubyte)
    /// \param label объект типа Matrix для заполнения
    inline void parse_mnist_label(const std::string &label_filename, Matrix &label) {
        std::ifstream ifs(label_filename.c_str(), std::ios::in | std::ios::binary);

        if (ifs.bad() || ifs.fail()) {
            throw internal::except::xs_error("[inline void parse_mnist_label] Error while opening file");
        }

        uint32_t magic_number, num_items;

        ifs.read(reinterpret_cast<char *>(&magic_number), 4);
        ifs.read(reinterpret_cast<char *>(&num_items), 4);

        if (internal::is_little_endian()) {
            internal::reverse_endian(&magic_number);
            internal::reverse_endian(&num_items);
        }

        if (magic_number != 0x00000801 || num_items <= 0) {
            throw internal::except::xs_error("[inline void parse_mnist_label] MNIST label-file format unknown");
        }

        label.resize(10, num_items);
        label.setZero();

        for (uint32_t i = 0; i < num_items; i++) {
            uint8_t label_;
            Scalar *arr = label.col(i).data();

            ifs.read(reinterpret_cast<char *>(&label_), 1);

            const uint8_t index = static_cast<uint8_t>(label_);

            arr[index] = 1;
        }
    }

    /// \brief Чтение изображений MNIST из файла
    /// \warning параметр label_filename должен содержать полный путь до файла
    /// \todo обозначить описание __x_padding__, __y_padding__
    /// \param image_filename расположение файла
    /// \param images train_data, train_label, etc
    /// \param scale_min минимальное значение масштабирования
    /// \param scale_max максимальное значение масштабирования
    /// \param x_padding
    /// \param y_padding
    inline void parse_mnist_image(const std::string &image_filename,
                                  Matrix &images,
                                  const Scalar &scale_min,
                                  const Scalar &scale_max,
                                  const int x_padding,
                                  const int y_padding) {
        // check param
        if (scale_min >= scale_max) {
            throw internal::except::xs_error(
                    "[inline void parse_mnist_image] \"scale max\" must be greater then \"scale min\"");
        }

        if (x_padding < 0 || y_padding < 0) {
            throw internal::except::xs_error("[inline void parse_mnist_image] \"padding\" must be greater then zero");
        }

        std::ifstream ifs(image_filename, std::ios::in | std::ios::binary);

        if (ifs.bad() || ifs.fail()) {
            throw internal::except::xs_error("[inline void parse_mnist_label] Error while opening file");
        }

        internal::mnist_header header;
        internal::parse_mnist_header(ifs, header);

        const int w = header.num_cols + 2 * x_padding;
        const int h = header.num_rows + 2 * y_padding;

        images.resize(w * h, header.num_items);

        for (uint32_t i = 0; i < header.num_items; i++) {
            Eigen::VectorXd image;

            internal::parse_mnist_image(ifs,
                                        header,

                                        w,
                                        h,

                                        scale_min,
                                        scale_max,
                                        x_padding,
                                        y_padding,

                                        image);

            images.col(i).noalias() = image;
        }
    }
} // end dataset

#endif //XSDNN_INCLUDE_MNIST_IO_H
