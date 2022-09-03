//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

TEST(fullyconnected, init){
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(10, 2, true);
    std::vector<Scalar> init_params = {0.0, 0.0};
    RNG rng(1);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> fc_layer_params = fc_layer->get_parametrs();

    EXPECT_EQ(fc_layer_params.size(), 10 * 2 + 2);
}

TEST(fullyconnected, init_without_bias){
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(10, 2, false);
    std::vector<Scalar> init_params = {0.0, 0.0};
    RNG rng(1);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> fc_layer_params = fc_layer->get_parametrs();

    EXPECT_EQ(fc_layer_params.size(), 10 * 2);
}

TEST(fullyconnected, grad){
    const int in_size = 1000;
    const int out_size = 500;

    Matrix train_image(in_size, 1); train_image.setRandom(); train_image *= 3.14;
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(in_size, out_size);

    std::vector<Scalar> init_param = {0.0, 1.0 / (1000.0 + 500.0)};
    RNG rng(1);
    fc_layer->init(init_param, rng);

    const int n = 100;
    for (int i = 0; i < n; i++)
    {
        const int in_pos_   = static_cast<int>(internal::random::set_uniform_random(rng, 0, in_size));
        const int out_pos_  = static_cast<int>(internal::random::set_uniform_random(rng, 0, out_size));

        Scalar    num_grad  = internal::debug::numerical_gradient(fc_layer,
                                                 train_image,
                                                 in_pos_,
                                                 out_pos_);

        Scalar    ana_grad  = internal::debug::analytical_gradient(fc_layer,
                                                  train_image,
                                                  in_pos_,
                                                  out_pos_,
                                                  out_size);

        EXPECT_NEAR(num_grad, ana_grad, sqrt_epsilon);
    }
}

TEST(fullyconnected, save){
    std::string filename = "fullyconnected.save";
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(1000, 1000, true);
    std::vector<Scalar> init_params = {0.0, 1.0};
    RNG rng(42);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> write   = fc_layer->get_parametrs();
    internal::io::write_one_vector(write, filename);

    std::vector<Scalar> read = internal::io::read_vector(filename);

    EXPECT_TRUE(is_near_container(write, read, Scalar(1E-5)));
}

TEST(fullyconnected, save_no_bias){
    std::string filename = "fullyconnected.save_no_bias";
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(1000, 1000, false);
    std::vector<Scalar> init_params = {0.0, 1.0};
    RNG rng(42);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> write   = fc_layer->get_parametrs();
    internal::io::write_one_vector(write, filename);

    std::vector<Scalar> read = internal::io::read_vector(filename);

    EXPECT_TRUE(is_near_container(write, read, Scalar(0.1)));
}

TEST(fullyconnected, forward){
    Layer* fc_layer = new FullyConnected<init::Constant, activate::Identity>(3, 3, true);
    std::vector<Scalar> init_params = {1.0};
    RNG rng(1);
    fc_layer->init(init_params, rng);

    Matrix in_data(3, 3);
    in_data <<  1, 1, 1,
                1, 1, 1,
                1, 1, 1;

    fc_layer->forward(in_data);
    Matrix out_data = fc_layer->output();

    Scalar* out_arr = out_data.data();
    for (int i = 0; i < in_data.size(); i++)
    {
        EXPECT_DOUBLE_EQ(Scalar(4.0), out_arr[i]);
    }
}

TEST(fullyconnected, forward_without_bias){
    Layer* fc_layer = new FullyConnected<init::Constant, activate::Identity>(3, 3, false);
    std::vector<Scalar> init_params = {1.0};
    RNG rng(1);
    fc_layer->init(init_params, rng);

    Matrix in_data(3, 3);
    in_data <<  1, 1, 1,
                1, 1, 1,
                1, 1, 1;

    fc_layer->forward(in_data);
    Matrix out_data = fc_layer->output();

    Scalar* out_arr = out_data.data();
    for (int i = 0; i < in_data.size(); i++)
    {
        EXPECT_DOUBLE_EQ(Scalar(3.0), out_arr[i]);
    }
}

