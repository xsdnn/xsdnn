//
// Created by shuffle on 17.08.22.
//

#ifndef XSDNN_DROPOUT_TEST_H
#define XSDNN_DROPOUT_TEST_H

TEST(dropout, determenistic)
{
#ifdef DNN_NO_DTRMINIST
    Dropout dp_layer = Dropout<activate::Identity>(784, 0.85);
    dp_layer.train();

    Matrix in_data(784, 1);
    in_data.setOnes();

    const int n = 1000;

    for (int k = 0; k < n; k++)
    {
        dp_layer.forward(in_data);
        Matrix mask = dp_layer.mask();

        dp_layer.forward(in_data);
        Matrix mask2 = dp_layer.mask();

        Scalar* mask_arr    = mask.data();
        Scalar* mask_arr2   = mask2.data();

        std::vector<bool> tf_vector(mask.size());
        for(int i = 0; i < mask.size(); i++)
        {
            if (mask_arr[i] == mask_arr2[i]) tf_vector[i] = true;
            else tf_vector[i] = false;
        }

        bool tf_first = tf_vector[0];
        bool flag     = false;

        for (int i = 1; i < mask.size(); i++)
        {
            if (tf_first != tf_vector[i]) flag = true;
        }

        EXPECT_TRUE(flag);
    }
#else
    Dropout dp_layer = Dropout<activate::Identity>(784, 0.85);
    dp_layer.train();

    Matrix in_data(784, 1);
    in_data.setOnes();

    const int n = 1000;

    for (int k = 0; k < n; k++)
    {
        dp_layer.forward(in_data);
        Matrix mask = dp_layer.mask();

        dp_layer.forward(in_data);
        Matrix mask2 = dp_layer.mask();

        Scalar* mask_arr    = mask.data();
        Scalar* mask_arr2   = mask2.data();

        for (int i = 0; i < mask.size(); i++)
        {
            EXPECT_DOUBLE_EQ(mask_arr[i], mask_arr2[i]);
        }
    }
#endif
}


#endif //XSDNN_DROPOUT_TEST_H
