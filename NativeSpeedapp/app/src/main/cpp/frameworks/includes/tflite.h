#include <iostream>
#include <vector>

using namespace std;

#ifndef TFLITE_H_
#define TFLITE_H_

namespace tflite_model {

    double  estimate_fps(char *path_model, vector<string> paths_images,
        int height, int width, int n_channels, bool torch_input_mode);
}

#endif