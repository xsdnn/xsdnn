//
// Created by shuffle on 06.06.22.
//

#ifndef XSDNN_INCLUDE_INOUT_H
#define XSDNN_INCLUDE_INOUT_H
# include <iostream>
# include <filesystem>
# include <iterator>
# include <vector>
# include <fstream>

namespace fs = std::filesystem;

namespace internal
{
    /// Создание директории при сохранении сетки
    /// \param directory_name название директории
    void create_directory(const std::string& directory_name)
    {
        fs::current_path("../xsDNN-models");
        if (fs::create_directory(directory_name))
        {
            std::cout << "Directory " << "'../xsDNN-models/" << directory_name << "' created successful" << std::endl;
        }
        else
        {
            std::cout << "Directory " << "'../xsDNN-models/" << directory_name << "' already established" << std::endl;
        }
    }

    /// Запись 1-D вектора в файл
    /// \param vec вектор
    /// \param filename название файла
    inline void write_one_vector(const std::vector<Scalar>& vec, std::string& filename)
    {
        // open or create file and write/rewrite into them
        std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
        if (ofs.fail())
        {
            throw std::runtime_error("[void write_one_vector] Error when opening file");
        }

        std::ostream_iterator<char> osi(ofs);
        const char* begin_byte = reinterpret_cast<const char*>(&vec[0]);
        const char* end_byte = begin_byte + vec.size() * sizeof(Scalar);
        std::copy(begin_byte, end_byte, osi);
        // std::cout << "File was successful wrote" << std::endl;
    }

    /// Запись n векторо в файл. n - кол-во слоев в сети.
    /// \param folder директория сохранения
    /// \param filename название файла с сеткой
    /// \param params 2-D вектор параметров
    inline void write_vector(std::string& folder, const std::string& filename,
                      std::vector< std::vector<Scalar> >& params)
    {
        const unsigned long nlayer = params.size();
        std::string folder_ = folder;
        for (unsigned long i = 0; i < nlayer; i++)
        {
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
    inline std::vector<Scalar> read_vector(const std::string& filename)
    {

        std::ifstream ifs(filename.c_str(), std::ios::in | std::ifstream::binary);
        if (ifs.fail())
            throw std::runtime_error("Error while opening file");

        std::vector<char> buffer;
        std::istreambuf_iterator<char> iter(ifs);
        std::istreambuf_iterator<char> end;
        std::copy(iter, end, std::back_inserter(buffer));
        std::vector<Scalar> vec(buffer.size() / sizeof(Scalar));
        std::copy(&buffer[0], &buffer[0] + buffer.size(), reinterpret_cast<char*>(&vec[0]));
        return vec;
    }

    /// Чтение параметров сети из бинарного файла модели
    /// \param folder папка с моделью
    /// \param filename название файла с моделью
    /// \param nlayer кол-во слоев в сети
    /// \return матрицу параметров
    inline std::vector< std::vector<Scalar> > read_parameter(
            const std::string& folder,
            const std::string& filename,
            const int& nlayer
            )
    {
        std::vector< std::vector<Scalar> > params;
        params.reserve(nlayer);

        std::string folder_ = folder;

        for (int i = 0; i < nlayer; i++)
        {
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
    inline void write_map(const std::string& filename, const std::map<std::string, int>& map)
    {
        if (map.empty())
            return;

        std::ofstream ofs(filename.c_str(), std::ios::out);
        if (ofs.fail())
            throw std::runtime_error("[void write_map] Error while opening file");

        for (std::map<std::string, int>::const_iterator it = map.begin(); it != map.end(); it++)
        {
            ofs << it->first << "=" << it->second << std::endl;
        }
    }

    /// Заполнение словаря с параметрами сетки
    /// \param filename название сетки
    /// \param map словарь
    inline void read_map(const std::string& filename, std::map<std::string, int>& map)
    {
        std::ifstream ifs(filename.c_str(), std::ios::in);

        if (ifs.fail())
            throw std::invalid_argument("[inline void read_map] Error when opening file.");

        map.clear();
        std::string buffer;

        while (std::getline(ifs, buffer))
        {
            unsigned long sep = buffer.find('=');

            if (sep == std::string::npos)
                throw std::invalid_argument("[inline void read_map] Error when reading file.");

            std::string key = buffer.substr(0, sep);
            std::string value = buffer.substr(sep + 1, buffer.length() - sep - 1);
            map[key] = std::stoi(value);
        }
    }
}


#endif //XSDNN_INCLUDE_INOUT_H
