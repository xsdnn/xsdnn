//
// Created by shuffle on 17.08.22.
//

// TODO: написать проверку градиента

TEST(fullyconnected, init)
{
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(10, 2, true);
    std::vector<Scalar> init_params = {0.0, 0.0};
    RNG rng(1);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> fc_layer_params = fc_layer->get_parametrs();

    EXPECT_EQ(fc_layer_params.size(), 10 * 2 + 2);
}

TEST(fullyconnected, init_without_bias)
{
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(10, 2, false);
    std::vector<Scalar> init_params = {0.0, 0.0};
    RNG rng(1);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> fc_layer_params = fc_layer->get_parametrs();

    EXPECT_EQ(fc_layer_params.size(), 10 * 2);
}

TEST(fullyconnected, save)
{
    std::string filename = "fullyconnected.save";
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(1000, 1000, true);
    std::vector<Scalar> init_params = {0.0, 1.0};
    RNG rng(42);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> write   = fc_layer->get_parametrs();
    internal::write_one_vector(write, filename);

    std::vector<Scalar> read = internal::read_vector(filename);

    EXPECT_TRUE(is_near_container(write, read, Scalar(0.1)));
}

TEST(fullyconnected, save_no_bias)
{
    std::string filename = "fullyconnected.save_no_bias";
    Layer* fc_layer = new FullyConnected<init::Normal, activate::Identity>(1000, 1000, false);
    std::vector<Scalar> init_params = {0.0, 1.0};
    RNG rng(42);

    fc_layer->init(init_params, rng);
    std::vector<Scalar> write   = fc_layer->get_parametrs();
    internal::write_one_vector(write, filename);

    std::vector<Scalar> read = internal::read_vector(filename);

    EXPECT_TRUE(is_near_container(write, read, Scalar(0.1)));
}

TEST(fullyconnected, forward)
{
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

TEST(fullyconnected, forward_without_bias)
{
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

