# How to?

## Процесс добавления слоя \ активации для разработчика:

1. Объявить класс в header-файлах в include.
2. Имплементировать в *src/layers/layer_name.cc* .
   1. Context - паттерн для хранения параметров слоя. 
   2. Kernel - паттерн, управляющий ядром слоя. Метод `compute` реализуется через
   виртуальную перегрузку метода 
   `virtual void compute(core::OpContext& ctx, 
   /*You're Parameters Holder (see params)*/& p) {}` . Имплементируется в `src/core/kernel` .
   
3. Зарегистрировать слой для класса network. Для этого использовать 
макрос `XS_REGISTER_LAYER_FOR_NET` в файле *src/layers/layer_register.cc* .
4. TODO: дополнить инструкцией по сериализации.