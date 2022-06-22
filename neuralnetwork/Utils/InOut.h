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
}


#endif //XSDNN_INCLUDE_INOUT_H
