//
// Created by shuffle on 24.07.22.
//

///
/// Тестирование генерации распределения Бернулли для маски Dropout слоя
///

// #####################################################################
// #                         TEST COMPLETED                            #
// #####################################################################

/*
 *  Microsoft test start...
    Use CTRL-Z to bypass data entry and run using default values.
    Enter an integer value for t distribution (where 0 <= t): 1
    Enter a double value for p distribution (where 0.0 <= p <= 1.0): 0.8
    Enter an integer value for a sample count: 100

    p == 0.8
    t == 1
    Histogram for 100 samples:
        0 :::::::::::::::::::::::::
        1 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    My test start...
    Use CTRL-Z to bypass data entry and run using default values.
    Enter a double value for p distribution (where 0.0 <= p <= 1.0): 0.8
    Enter an integer value for a sample count: 100

    p == 0.8
    Histogram for 100 samples:
        0 :::::::::::::::::::
        1 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

# include "../neuralnetwork/DNN.h"
# include <random>
# include <iostream>
# include <iomanip>
# include <string>
# include <map>

void microsoft_test(const int t, const double p, const int& s) {

    // uncomment to use a non-deterministic seed
//        std::random_device rd;
//        std::mt19937 gen(rd());

    std::mt19937 gen(1729);

    std::binomial_distribution<> distr(t, p);

    std::cout << std::endl;
    std::cout << "p == " << distr.p() << std::endl;
    std::cout << "t == " << distr.t() << std::endl;

    // generate the distribution as a histogram
    std::map<int, int> histogram;
    for (int i = 0; i < s; ++i) {
        ++histogram[distr(gen)];
    }

    // print results
    std::cout << "Histogram for " << s << " samples:" << std::endl;
    for (const auto& elem : histogram) {
        std::cout << std::setw(5) << elem.first << ' ' << std::string(elem.second, ':') << std::endl;
    }
    std::cout << std::endl;
}

void my_test(const Scalar& p, const int& s)
{
    RNG rng(42);

    std::cout << std::endl;
    std::cout << "p == " << p << std::endl;

    std::map<int, int> histogram;
    for (int i = 0; i < s; ++i) {
        ++histogram[static_cast<int>(internal::bernoulli(p, rng))];
    }

    // print results
    std::cout << "Histogram for " << s << " samples:" << std::endl;
    for (const auto& elem : histogram) {
        std::cout << std::setw(5) << elem.first << ' ' << std::string(elem.second, ':') << std::endl;
    }
    std::cout << std::endl;
}

void microsoft_test_start()
{
    int    t_dist = 1;
    Scalar p_dist = 0.5;
    int    samples = 100;

    std::cout << "Use CTRL-Z to bypass data entry and run using default values." << std::endl;
    std::cout << "Enter an integer value for t distribution (where 0 <= t): ";
    std::cin >> t_dist;
    std::cout << "Enter a double value for p distribution (where 0.0 <= p <= 1.0): ";
    std::cin >> p_dist;
    std::cout << "Enter an integer value for a sample count: ";
    std::cin >> samples;

    microsoft_test(t_dist, p_dist, samples);
}

void my_test_start()
{
    Scalar p_dist = 0.5;
    int    samples = 100;

    std::cout << "Use CTRL-Z to bypass data entry and run using default values." << std::endl;
    std::cout << "Enter a double value for p distribution (where 0.0 <= p <= 1.0): ";
    std::cin >> p_dist;
    std::cout << "Enter an integer value for a sample count: ";
    std::cin >> samples;

    my_test(p_dist, samples);
}
int main()
{
    std::cout << "Microsoft test start..." << std::endl;
    microsoft_test_start();

    std::cout << "My test start..." << std::endl;
    my_test_start();
}