//
// Created by shuffle on 30.07.22.
//

# include "unistd.h"
# include "../neuralnetwork/xsDNN.h"

int main() {
    internal::ProgressBar disp(1000);

    for (int i = 0; i < 20; i++)
    {
        disp += 50;
        sleep(1);
    }

}