#include <iostream>
#include <vector>

using namespace std;

#ifndef TORCH_H_
#define TORCH_H_

namespace torch_model {

    double  estimate_fps(char *path_model, vector<string> paths_images,
        int height, int width, int n_channels);
}

#endif