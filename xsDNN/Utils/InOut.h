//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_INCLUDE_INOUT_H
#define XSDNN_INCLUDE_INOUT_H

# include <iostream>
# include <filesystem>
# include <iterator>
# include <vector>
# include <fstream>
# include "../Config.h"
#include "Except.h"

# include <chrono>

namespace fs = std::filesystem;

namespace internal {
    namespace io {
        /// Создание директории при сохранении сетки
        /// \param directory_name название директории
        void create_directory(const std::string &directory_name) {
            fs::current_path("./");
            if (fs::create_directory(directory_name)) {
                std::cout << "Directory " << directory_name << "' created successful" << std::endl;
            } else {
                std::cout << "Directory " << directory_name << "' already established" << std::endl;
            }
        }

        /// Запись 1-D вектора в файл
        /// \param vec вектор
        /// \param filename название файла
        inline void write_one_vector(const std::vector<Scalar> &vec, std::string &filename) {
            // open or create file and write/rewrite into them
            std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
            if (ofs.fail()) {
                throw internal::except::xs_error("[void write_one_vector] Error when opening file");
            }

            std::ostream_iterator<char> osi(ofs);
            const char *begin_byte = reinterpret_cast<const char *>(&vec[0]);
            const char *end_byte = begin_byte + vec.size() * sizeof(Scalar);
            std::copy(begin_byte, end_byte, osi);
            // std::cout << "File was successful wrote" << std::endl;
        }

        /// Запись n векторо в файл. n - кол-во слоев в сети.
        /// \param folder директория сохранения
        /// \param filename название файла с сеткой
        /// \param params 2-D вектор параметров
        inline void write_vector(const std::string &folder, const std::string &filename,
                                 std::vector<std::vector<Scalar> > &params) {
            const unsigned long nlayer = params.size();
            std::string folder_ = folder;
            for (unsigned long i = 0; i < nlayer; i++) {
                folder_.append("/");
                folder_.append(filename);
                folder_.append(std::to_string(i));
                write_one_vector(params[i], folder_);
                folder_ = folder;
            }
        }

        /// Чтение параметров для одного слоя
        /// \param filename название файла
        /// \return вектор параметров для одного слоя
        inline std::vector<Scalar> read_vector(const std::string &filename) {

            std::ifstream ifs(filename.c_str(), std::ios::in | std::ifstream::binary);
            if (ifs.fail())
                throw internal::except::xs_error("[inline read_vector] Error while opening file");

            std::vector<char> buffer;
            std::istreambuf_iterator<char> iter(ifs);
            std::istreambuf_iterator<char> end;
            std::copy(iter, end, std::back_inserter(buffer));
            std::vector<Scalar> vec(buffer.size() / sizeof(Scalar));
            std::copy(&buffer[0], &buffer[0] + buffer.size(), reinterpret_cast<char *>(&vec[0]));
            return vec;
        }

        /// Чтение параметров сети из бинарного файла модели
        /// \param folder папка с моделью
        /// \param filename название файла с моделью
        /// \param nlayer кол-во слоев в сети
        /// \return матрицу параметров
        inline std::vector<std::vector<Scalar> > read_parameter(
                const std::string &folder,
                const std::string &filename,
                const int &nlayer
        ) {
            std::vector<std::vector<Scalar> > params;
            params.reserve(nlayer);

            std::string folder_ = folder;

            for (int i = 0; i < nlayer; i++) {
                folder_.append("/");
                folder_.append(filename);
                folder_.append(std::to_string(i));
                params.push_back(read_vector(folder_));
                folder_ = folder;
            }

            return params;
        }

        /// Запись файла с информацией о слоях в сети
        /// \param filename название файла с сеткой
        /// \param map словарь, в которой хранится информация о слое
        inline void write_map(const std::string &filename, const std::map<std::string, Scalar> &map) {
            if (map.empty())
                return;

            std::ofstream ofs(filename.c_str(), std::ios::out);
            if (ofs.fail())
                throw internal::except::xs_error("[void write_map] Error while opening file");

            for (std::map<std::string, Scalar>::const_iterator it = map.begin(); it != map.end(); it++) {
                ofs << it->first << "=" << it->second << std::endl;
            }
        }

        /// Заполнение словаря с параметрами сетки
        /// \param filename название сетки
        /// \param map словарь
        inline void read_map(const std::string &filename, std::map<std::string, Scalar> &map) {
            std::ifstream ifs(filename, std::ios::in);

            if (ifs.fail())
                throw internal::except::xs_error("[inline void read_map] Error when opening file.");

            map.clear();
            std::string buffer;

            while (std::getline(ifs, buffer)) {
                unsigned long sep = buffer.find('=');

                if (sep == std::string::npos)
                    throw internal::except::xs_error("[inline void read_map] Error when reading file.");

                std::string key = buffer.substr(0, sep);
                std::string value = buffer.substr(sep + 1, buffer.length() - sep - 1);
                map[key] = std::stoi(value);
            }
        }
    } // end namespace io

    namespace display {
        class Timer {
        private:
            std::chrono::high_resolution_clock::time_point t1, t2;

        public:
            explicit Timer() : t1(std::chrono::high_resolution_clock::now()) {}

            Scalar elapced() {
                return std::chrono::duration_cast<std::chrono::duration<Scalar>>(

                        std::chrono::high_resolution_clock::now() - t1

                ).count();
            }

            void restart() { t1 = std::chrono::high_resolution_clock::now(); }

            void start() { t1 = std::chrono::high_resolution_clock::now(); }

            void stop() { t2 = std::chrono::high_resolution_clock::now(); }

            Scalar total() {
                this->stop();
                return std::chrono::duration_cast<std::chrono::duration<Scalar>>(

                        t2 - t1

                ).count();
            }

            ~Timer() {}
        };

        class ProgressBar {
        private:
            typedef unsigned int IntType;

            IntType _num_total;            // размер всего тренировочного / тестового набора
            IntType _num_succes;           // кол-во объектов, прошедших обучение сети
            IntType _next_tic_count;
            IntType _tic;


            std::ostream &out;

            void display_update() {
                IntType update_needed = static_cast<IntType>(

                        (static_cast<double>(_num_succes) / _num_total) * 50.0

                );

                do {
                    out << "*" << std::flush;
                } while (++_tic < update_needed);

                _next_tic_count = static_cast<IntType>((_tic / 50.0) * _num_total);
                if (_num_succes == _num_total) {
                    if (_tic < 51) out << '*';
                    out << std::endl;
                }
            }

        public:
            explicit ProgressBar(
                    IntType num_total,
                    std::ostream &os = std::cout
            ) : out(os) { this->restart(num_total); }

            void restart(IntType num_total) {
                _num_succes = _next_tic_count = _tic = 0;
                _num_total = num_total;

                out << "\n" << "0%   10   20   30   40   50   60   70   80   90   100%\n"
                    << "" << "|----|----|----|----|----|----|----|----|----|----|"
                    << std::endl << "";
            }

            IntType operator+=(IntType increment) {
                if ((_num_succes += increment) >= _next_tic_count) { this->display_update(); }
                return _num_succes;
            }
        };
    } // end namespace display
} // end namespace internal


#endif //XSDNN_INCLUDE_INOUT_H
