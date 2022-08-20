//
// Created by shuffle on 17.08.22.
//
#ifndef XSDNN_DROPOUT_TEST_H
#define XSDNN_DROPOUT_TEST_H

TEST(dropout, determenistic){
#ifdef DNN_NO_DTRMINIST
    Dropout dp_layer = Dropout<activate::Identity>(784, 0.85);
    dp_layer.train();

    Matrix in_data(784, 4);
    in_data.setOnes();

    const int n = 1000;

    for (int k = 0; k < n; k++)
    {
        dp_layer.forward(in_data);
        Matrix mask = dp_layer.mask();

        dp_layer.forward(in_data);
        Matrix mask2 = dp_layer.mask();

        EXPECT_TRUE(is_different_container(mask, mask2));
    }
#else
    Dropout dp_layer = Dropout<activate::Identity>(784, 0.85);
    dp_layer.train();

    Matrix in_data(784, 4);
    in_data.setOnes();

    const int n = 1000;

    for (int k = 0; k < n; k++)
    {
        dp_layer.forward(in_data);
        Matrix mask = dp_layer.mask();

        dp_layer.forward(in_data);
        Matrix mask2 = dp_layer.mask();

        EXPECT_TRUE(is_equal_container(mask, mask2));
    }
#endif
}

TEST(dropout, fully_net)
{
    Matrix train_image(784, 16), train_label(784, 16);
    generate_sinus_data(train_image, train_label, 3.14);

    NeuralNetwork net;

    using DP = Dropout<activate::Identity>;

    net     << new DP(784, 0.3)
            << new DP(784, 0.4);

    Output* criterion = new MSELoss();
    net.set_output(criterion);
    SGD opt;

    EXPECT_TRUE(net.fit(opt, train_image, train_label, 16, 5, 42));
}


#endif //XSDNN_DROPOUT_TEST_H
