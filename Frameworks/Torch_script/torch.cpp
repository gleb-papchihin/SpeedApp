
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/script.h>
#include <iostream>
#include <ctime>

using namespace std;


namespace torch_model{

    void            load_torchscript_module(char *path, torch::jit::script::Module &model) {
        model = torch::jit::load(path);
    }

    void            resize_image(cv::Mat &image, cv::Mat &resized, int height, int width) {
        cv::resize(image, resized, cv::Size(width, height));
    }

    void            load_image(string path, cv::Mat &image, int height, 
        int width, int n_channels) {
        image = cv::imread(path);

        // Resize image
        resize_image(image, image, height, width);

        // Change colors
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Convert to float type
        if (n_channels == 3)
            image.convertTo(image, CV_32FC3);
        else if (n_channels == 1)
            image.convertTo(image, CV_32FC1);
        else
            throw invalid_argument("Torch model: Unsupported number of channels");
    }

    double          get_mean_time(vector<double> &exec_time) {
        
        double mean_time;

        for (size_t i = 0; i < exec_time.size(); i++)
            mean_time += exec_time.at(i);

        mean_time /= (int) exec_time.size();
        return mean_time;
    }

    double          convert_mean_time_to_fps(double mean_time) {
        double fps = (double) (1.0 / mean_time);
        return fps;
    }

    void            convert_cv_to_tensor(cv::Mat &image, torch::Tensor &tensor) {
        tensor = torch::from_blob(image.ptr<float>(0), 
            {image.rows, image.cols, image.channels()});
        tensor = tensor.permute({ 2,0,1 });
        tensor = tensor.toType(c10::kFloat);
        tensor = tensor.unsqueeze(0);
    }

    vector<string>  split_string_by_symbol(char *str, char symbol) {
        vector<string>  splits;
        string          buffer;

        buffer = "";
        for(int i = 0; str[i] != '\0'; i++) {
            if (str[i] == symbol) {
                splits.push_back(buffer);
                buffer = "";
            }
            else {
                buffer += str[i];
            }
        }

        if (buffer != "") {
            splits.push_back(buffer);
        }

        return splits;
    }

    double          estimate_fps(char *path_model, vector<string> paths_images, 
        int height, int width, int n_channels) {

        cv::Mat                         image;
        int                             image_size;
        double                          mean_time;
        double                          estimated_fps;
        double                          time;
        clock_t                         start;
        clock_t                         stop;
        vector<double>                  exec_time;
        torch::jit::script::Module      model;
        string                          path_image;
        torch::Tensor                   input_tensor;

        // Load model
        load_torchscript_module(path_model, model);

        for (size_t i = 0; i < paths_images.size(); i++) {

            path_image = paths_images[i];

            // Load image
            load_image(path_image, image, height, width, n_channels);

            // Convert cv to tensor
            convert_cv_to_tensor(image, input_tensor);

            // Load tensor to vector
            vector<torch::jit::IValue> inputs = {input_tensor};

            // Process image
            start = clock();
            model.forward(inputs);
            stop = clock();

            time = (double)((stop - start) / CLOCKS_PER_SEC);
            exec_time.push_back(time);
        }

        mean_time = get_mean_time(exec_time);
        estimated_fps = convert_mean_time_to_fps(mean_time);

        return estimated_fps;
    }

    double          estimate_fps(char *path_model, char *paths_images, 
        int height, int width, int n_channels) {

        cv::Mat                         image;
        vector<string>                  paths_images_vec;
        int                             image_size;
        double                          mean_time;
        double                          estimated_fps;
        double                          time;
        clock_t                         start;
        clock_t                         stop;
        vector<double>                  exec_time;
        torch::jit::script::Module      model;
        string                          path_image;
        torch::Tensor                   input_tensor;

        // convert string to vector
        paths_images_vec = split_string_by_symbol(paths_images, ';');

        // Load model
        load_torchscript_module(path_model, model);

        for (size_t i = 0; i < paths_images_vec.size(); i++) {

            path_image = paths_images_vec[i];

            // Load image
            load_image(path_image, image, height, width, n_channels);

            // Convert cv to tensor
            convert_cv_to_tensor(image, input_tensor);

            // Load tensor to vector
            vector<torch::jit::IValue> inputs = {input_tensor};

            // Process image
            start = clock();
            model.forward(inputs);
            stop = clock();

            time = (double)((stop - start) / CLOCKS_PER_SEC);
            exec_time.push_back(time);
        }

        mean_time = get_mean_time(exec_time);
        estimated_fps = convert_mean_time_to_fps(mean_time);

        return estimated_fps;
    }
}

/*
int main() {
    char*           path_model;
    string          path_image;
    vector<string>  paths_images;

    path_model = (char *)"/home/gleb/CMakeTest/torchscript/yolov5s.torchscript.pt";
    path_image = "/home/gleb/CMakeTest/torchscript/gleb.jpg";

    for (int i = 0; i < 6; i++)
        paths_images.push_back(path_image);

    cout << "FPS: " << fixed << torch_model::estimate_fps(path_model, paths_images, 640, 640, 3) << endl;

    return 0;
}
*/