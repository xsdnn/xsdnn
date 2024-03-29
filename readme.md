# How to?

## Процесс добавления слоя \ активации для разработчика:

**IMPORTANT:** порядок следования концептов при инициализации слоя - `{data, weight, bias}`.

1. Объявить класс в header-файлах в include.
2. Имплементировать в *src/layers/layer_name.cc* .
   1. Context - паттерн для хранения параметров слоя. 
   2. Kernel - паттерн, управляющий ядром слоя. Метод `compute` реализуется через
   виртуальную перегрузку метода 
   `virtual void compute(core::OpContext& ctx, 
   /*You're Parameters Holder (see params)*/& p) {}` . Имплементируется в `src/core/kernel` .
   3. **Внимание!** Возвращаемое значение метода `layer_type` должно полностью совпадать с названием слоя, иначе 
   сериализация будет не доступна для данного слоя.
3. Добавьте заголовочный файл класса в `include/layer/layer.h`
3. Для добавления сериализации сделайте другом класса слоя класс cerial. `friend class cerial;`
4. В классе *cerial* по пути `include/serializer/cerial.h` имплементируйте методы `serialize/deserialize`
согласно инструкции добавления.
5. Зарегистрируйте слой в файле `src/layer/layer_register.cc`, добавив макросы 
`XS_LAYER_SAVE_INTERNAL_REGISTER(layer_typename)` и `XS_LAYER_LOAD_INTERNAL_REGISTER(layer_typename)`.


## Процесс добавления инициализатора весов \ смещений для разработчика

1. Объявить класс инициализации и унаследовать его от *function* по пути 
`include/utils/weight_init.h`.
2. Имплентировать метод `fill` по пути `src/utils/weight_init.cc`

****

# Some note's

**ALWAYS ROW-MAJOR FORMAT**

1. В пространстве пользователя все данные, включая изображения и \ или любые другие типы данных должны быть представлены 
в виде одномерного вектора типа `xsdnn::mat_t`.
2. Тип `xsdnn::tensor_t` в пространстве пользователя используется как контейнер для нескольких объектов, например, 
изображений. 
3. В пространстве пользователя при работе с графовым представлением нейросети порядок входных данных описывается так:
``[sample][input_layer_id][dim]``.   
4. В пространстве пользователя при работе с графовым представлением нейросети порядок выходных данных описывается так:
``[sample][output_layer_id][dim]``.   
5. Поддерживается **только** формат `NCHW`.