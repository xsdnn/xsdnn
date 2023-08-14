# Inference Example

Интерфейс библиотеки схож с интерфейсом onnxruntime, за исключением сильного упрощения внутренних компонентов. 

# Загрузка сети из формата xs

```c++
/*
 * Process in data ...
 */

InfOptions opt;

opt.SetNetType(net_type::sequential);
opt.SetNumThreads(10);

InfSession session(opt);
session.Load(<path_to_xs_model>);
session.Run(std::vector<tensor>& in, std::vector<tensor> out);

/*
 * Process out data ...
 */
```

