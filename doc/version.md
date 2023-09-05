# Версионирование

## Критерии выпуска версии 1.0.0 (23-Q3)

### 1. Поддержка fwd-pass у слоев:
- [x] fully_connected
- [x] abs
- [x] acos
- [x] add
- [x] and
- [x] batch_norm2d
- [x] flatten
- [x] fully_connected
- [x] input
- [x] output
- [x] conv
- [x] max_pool
- [x] global_average_pool
- [x] reshape
- [ ] matmul
- [ ] mul
- [ ] hard_sigmoid
- [ ] hard_swish
- [ ] clip
- [ ] exp
- [ ] sigmoid
- [ ] transpose

### 2. Поддержка сериализации у слоев: 
- [x] fully_connected
- [x] abs
- [x] acos
- [x] add
- [x] and
- [x] batch_norm2d
- [x] flatten
- [x] fully_connected
- [x] input
- [x] output
- [x] conv
- [x] max_pool
- [x] global_average_pool
- [x] reshape
- [ ] matmul
- [ ] mul
- [ ] hard_sigmoid
- [ ] hard_swish
- [ ] clip
- [ ] exp
- [ ] sigmoid
- [ ] transpose


### 3. Рефакторинг скрипта сборки
- [x] Билд динамической библиотеки
- [x] Возможность установки в /usr/lib
- [x] Вывод summary информации о билде


# Критерии выпуска версии 1.1.0 (23-Q4)

### 1. Интеграция XNNPACK

### 2. Автоматический выбор размерности операции (1D, 2D) используя входной shape

# Критерии выпуска версии 1.2.0 (24-Q2)

### 1. Поддержка операций fp16 для x86_64 и arm64 для слоев:
- [ ] fully_connected
- [ ] batch_norm2d
- [ ] conv
- [ ] max_pool
- [ ] average_pool
- [ ] global_average_pool

### 2. Поддержка формата  NHWC

# Критерии выпуска версии 1.3.0 (24-Q2)

### 1. Настройка многопоточности

- [ ] Проанализировать необходимость параллелизма GEMM
- [ ] Имплементировать при необходимости