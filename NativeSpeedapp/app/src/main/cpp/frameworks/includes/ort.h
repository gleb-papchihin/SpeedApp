#include <iostream>
#include <vector>

using namespace std;

#ifndef ORT_H_
#define ORT_H_

namespace ort_model {

    double  estimate_fps(char *path_model, vector<string> paths_images,
        int height, int width, int n_channels, bool torch_input_mode);
}

#endif