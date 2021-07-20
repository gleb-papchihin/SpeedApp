
// Problem with version of GCC
// https://web-answers.ru/c/neopredelennaja-ssylka-na-process-std-cxx11-basic.html
#define _GLIBCXX_USE_CXX11_ABI 0

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <iostream>
#include <ctime>

using namespace std;

namespace tflite_model {

    int             get_index_in_cv_format(int x, int y, int c, 
        int height, int width, int n_channels) {
        // format is [1, height, width, n_channels]
        int index = (y * width * n_channels) + (x * n_channels) + c;
        return index;
    }

    int             get_index_in_torch_format(int x, int y, int c, 
        int height, int width, int n_channels) {
        // format is [1, n_channels, height, width]
        int index = (c * height * width) + (y * width) + x;
        return index;
    }

    void            fill_input_tensor_tf_mode(uint8_t* input, float* tensor,
        int height, int width, int n_channels) {
        // Shape of a torch input is [1, height, width, n_channels]

        int index;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < n_channels; c++) {

                    // get an index corresponding to flatten shape.
                    index = get_index_in_cv_format(x, y, c, 
                        height, width, n_channels);

                    // fill element
                    tensor[index] = (float) input[index];
                }
            }
        }
    }

    void            fill_input_tensor_torch_mode(uint8_t* input, float* tensor,
        int height, int width, int n_channels) {
        // Shape of a torch input is [1, n_channels, height, width]
        // Helpful if you convert pytorch model to tflite.

        int     torch_index;
        int     cv_index;

        for (int c = 0; c < n_channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                    // get an index corresponding to flatten shape.
                    torch_index = get_index_in_torch_format(x, y, c, 
                        height, width, n_channels);

                    // get an index corresponding to flatten shape.
                    cv_index = get_index_in_cv_format(x, y, c, 
                        height, width, n_channels);

                    // fill element
                    tensor[torch_index] = input[cv_index];
                }
            }
        }
    }

    void            fill_input_tensor(uint8_t* input, float* tensor,
        int height, int width, int n_channels, bool torch_input_mode) {

        if (torch_input_mode == true)
            fill_input_tensor_torch_mode(input, tensor, height, width, n_channels);
        else
            fill_input_tensor_tf_mode(input, tensor, height, width, n_channels);
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
        int height, int width, int n_channels, bool torch_input_mode) {

        int                                         input;
        double                                      mean_time;
        double                                      estimated_fps;
        double                                      time;
        clock_t                                     start;
        clock_t                                     stop;
        float*                                      input_tensor;
        uint8_t*                                    image_data;
        string                                      path_image;
        cv::Mat                                     image;
        vector<double>                              exec_time;
        unique_ptr<tflite::FlatBufferModel>         model;
        tflite::ops::builtin::BuiltinOpResolver     resolver;
        std::unique_ptr<tflite::Interpreter>        interpreter;

        // Load model.
        model = tflite::FlatBufferModel::BuildFromFile(path_model);

        // Check if the model was loaded successfully
        if (!model)
            throw invalid_argument("TF Lite model: Can not read a model. Check the path.");

        // build interpriter
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);

        // Check if the model was built
        if (!interpreter)
            throw runtime_error("TF Lite model: Can not build an interpreter :(");

        // Allocate buffer
        interpreter->AllocateTensors();

        // Input meta
        input = interpreter->inputs()[0];

        for (size_t i = 0; i < paths_images.size(); i++) {

            path_image = paths_images[i];

            // Load image.
            load_image(path_image, image, height, width, n_channels);
            image_data = image.ptr<uint8_t>(0);
            
            // Input tensor
            input_tensor = interpreter->typed_input_tensor<float>(input);

            // Fill input tensor
            fill_input_tensor(image_data, input_tensor, height, width, 
                n_channels, torch_input_mode);

            // Get execution time
            start = clock();
            interpreter->Invoke();        
            stop = clock();

            time = (double)((stop - start) / CLOCKS_PER_SEC);
            exec_time.push_back(time);
        }

        mean_time = get_mean_time(exec_time);
        estimated_fps = convert_mean_time_to_fps(mean_time);

        return estimated_fps;
    }

    double          estimate_fps(char *path_model, char *paths_images,
        int height, int width, int n_channels, bool torch_input_mode) {

        // In this case, paths_images has 
        // the format: path_image_0;path_image_1;...

        int                                         input;
        double                                      mean_time;
        double                                      estimated_fps;
        double                                      time;
        clock_t                                     start;
        clock_t                                     stop;
        float*                                      input_tensor;
        uint8_t*                                    image_data;
        string                                      path_image;
        cv::Mat                                     image;
        vector<string>                              paths_images_vec;
        vector<double>                              exec_time;
        unique_ptr<tflite::FlatBufferModel>         model;
        tflite::ops::builtin::BuiltinOpResolver     resolver;
        std::unique_ptr<tflite::Interpreter>        interpreter;

        // convert string to vector
        paths_images_vec = split_string_by_symbol(paths_images, ';');

        // Load model.
        model = tflite::FlatBufferModel::BuildFromFile(path_model);

        // Check if the model was loaded successfully
        if (!model)
            throw invalid_argument("TF Lite model: Can not read a model. Check the path.");

        // build interpriter
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);

        // Check if the model was built
        if (!interpreter)
            throw runtime_error("TF Lite model: Can not build an interpreter :(");

        // Allocate buffer
        interpreter->AllocateTensors();

        // Input meta
        input = interpreter->inputs()[0];

        for (size_t i = 0; i < paths_images_vec.size(); i++) {

            path_image = paths_images_vec[i];

            // Load image.
            load_image(path_image, image, height, width, n_channels);
            image_data = image.ptr<uint8_t>(0);
            
            // Input tensor
            input_tensor = interpreter->typed_input_tensor<float>(input);

            // Fill input tensor
            fill_input_tensor(image_data, input_tensor, height, width, 
                n_channels, torch_input_mode);

            // Get execution time
            start = clock();
            interpreter->Invoke();        
            stop = clock();

            time = (double)((stop - start) / CLOCKS_PER_SEC);
            exec_time.push_back(time);
        }

        mean_time = get_mean_time(exec_time);
        estimated_fps = convert_mean_time_to_fps(mean_time);

        return estimated_fps;
    }
}