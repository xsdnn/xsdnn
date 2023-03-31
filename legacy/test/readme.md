# Тесты - инструкция по добавлению

1. Каждая группа тестов - это отдельная единица трансляции :: *.cpp файл
2. После написания тестов - добавить точку входа в формате:

```c++
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

3. В `CMakeLists.txt` добавить цель и слинковать с `GTest::gtest_main`. Пример:

```cmake
add_executable(
        edge_node_test
        edge_node_test.cpp
)
target_link_libraries(
        edge_node_test
        GTest::gtest_main
)
```
