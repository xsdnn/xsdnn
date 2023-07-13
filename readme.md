# How to?

## Процесс добавления слоя \ активации для разработчика:

1. Объявить класс в header-файлах в include.
2. Имплементировать в *src/layers/layer_name.cc* .
   1. Context - паттерн для хранения параметров слоя. 
   2. Kernel - паттерн, управляющий ядром слоя. Метод `compute` реализуется через
   виртуальную перегрузку метода 
   `virtual void compute(core::OpContext& ctx, 
   /*You're Parameters Holder (see params)*/& p) {}` . Имплементируется в `src/core/kernel` .
   
3. **Зарегистрировать** слой для класса network. Для этого использовать 
макрос `XS_REGISTER_LAYER_FOR_NET` в файле *src/layers/layer_register.cc* .
4. TODO: дополнить инструкцией по сериализации.

## Процесс добавления инициализатора весов \ смещений для разработчика

1. Объявить класс инициализации и унаследовать его от *function* по пути 
`include/utils/weight_init.h`.
2. Имплентировать метод `fill` по пути `src/utils/weight_init.cc`
3. **Зарегистрировать** инициализатор весов \ смещений по пути `src/layers/layer.cc`. 
Для этого использовать макросы `XS_REGISTER_WEIGHT_INIT` \ `XS_REGISTER_BIAS_INIT`. 