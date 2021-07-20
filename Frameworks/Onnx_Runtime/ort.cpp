
// Problem with version of GCC
// https://web-answers.ru/c/neopredelennaja-ssylka-na-process-std-cxx11-basic.html
#define _GLIBCXX_USE_CXX11_ABI 0

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <ctime>

using namespace std;


namespace ort_model {
    
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

    double          estimate_fps (char *path_model, vector<string> paths_images, 
        int height, int width, int n_channels, bool torch_input_mode) {
        
        cv::Mat                             image;
        double                              mean_time;
        double                              estimated_fps;
        double                              time;
        int                                 image_size;
        uint8_t*                            image_data;
        vector<double>                      exec_time;
        clock_t                             start;
        clock_t                             stop;
        size_t                              n_input_nodes;
        size_t                              n_output_nodes;
        char*                               input_name;
        char*                               output_name;
        string                              path_image;
        Ort::Env                            env;
        Ort::SessionOptions                 session_options;
        Ort::AllocatorWithDefaultOptions    allocator;

        // Vector for converting opencv mat.
        image_size = height * width * n_channels;
        vector<float> input_flatten(image_size);

        // Set options.
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Init session
        Ort::Session session(env, path_model, session_options);

        // Get inputs names
        n_input_nodes = session.GetInputCount();
        vector<const char*> input_names(n_input_nodes);
        for (size_t i = 0; i < n_input_nodes; i++) {
            input_name = session.GetInputName(i, allocator);
            input_names[i] = input_name;
        }

        // Get outputs names
        n_output_nodes = session.GetOutputCount();
        vector<const char*> output_names(n_output_nodes);
        for (size_t i = 0; i < n_output_nodes; i++) {
            output_name = session.GetOutputName(i, allocator);
            output_names[i] = output_name;
        }

        // Get input shape
        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        for (size_t i = 0; i < paths_images.size(); i++) {
            
            path_image = paths_images[i];

            // load image
            load_image(path_image, image, height, width, n_channels);
            image_data = image.ptr<uint8_t>(0);

            // fill buffer from image
            fill_input_tensor(image_data, input_flatten.data(), height, 
                width, n_channels, torch_input_mode);

            // fill tensor from buffer
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                input_flatten.data(), image_size, input_shape.data(), input_shape.size());

            start = clock();
            session.Run( Ort::RunOptions{ nullptr }, 
                input_names.data(), &input_tensor, 1,
                output_names.data(), 1);
            stop = clock();

            time = (double)((stop - start) / CLOCKS_PER_SEC);
            exec_time.push_back(time);
        }

        mean_time = get_mean_time(exec_time);
        estimated_fps = convert_mean_time_to_fps(mean_time);

        return estimated_fps;
    }

    double          estimate_fps (char *path_model, char *paths_images, 
        int height, int width, int n_channels, bool torch_input_mode) {
        
        cv::Mat                             image;
        double                              mean_time;
        double                              estimated_fps;
        vector<string>                      paths_images_vec;
        double                              time;
        int                                 image_size;
        uint8_t*                            image_data;
        vector<double>                      exec_time;
        clock_t                             start;
        clock_t                             stop;
        size_t                              n_input_nodes;
        size_t                              n_output_nodes;
        char*                               input_name;
        char*                               output_name;
        string                              path_image;
        Ort::Env                            env;
        Ort::SessionOptions                 session_options;
        Ort::AllocatorWithDefaultOptions    allocator;

        // convert string to vector
        paths_images_vec = split_string_by_symbol(paths_images, ';');

        // Vector for converting opencv mat.
        image_size = height * width * n_channels;
        vector<float> input_flatten(image_size);

        // Set options.
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Init session
        Ort::Session session(env, path_model, session_options);

        // Get inputs names
        n_input_nodes = session.GetInputCount();
        vector<const char*> input_names(n_input_nodes);
        for (size_t i = 0; i < n_input_nodes; i++) {
            input_name = session.GetInputName(i, allocator);
            input_names[i] = input_name;
        }

        // Get outputs names
        n_output_nodes = session.GetOutputCount();
        vector<const char*> output_names(n_output_nodes);
        for (size_t i = 0; i < n_output_nodes; i++) {
            output_name = session.GetOutputName(i, allocator);
            output_names[i] = output_name;
        }

        // Get input shape
        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        for (size_t i = 0; i < paths_images_vec.size(); i++) {
            
            path_image = paths_images_vec[i];

            // load image
            load_image(path_image, image, height, width, n_channels);
            image_data = image.ptr<uint8_t>(0);

            // fill buffer from image
            fill_input_tensor(image_data, input_flatten.data(), height, 
                width, n_channels, torch_input_mode);

            // fill tensor from buffer
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                input_flatten.data(), image_size, input_shape.data(), input_shape.size());

            start = clock();
            session.Run( Ort::RunOptions{ nullptr }, 
                input_names.data(), &input_tensor, 1,
                output_names.data(), 1);
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

    path_model = (char *)"/home/gleb/CMakeTest/ort/yolov5s.all.ort";
    path_image = "/home/gleb/CMakeTest/torchscript/gleb.jpg";

    for (int i = 0; i < 6; i++)
        paths_images.push_back(path_image);

    cout << "FPS: " << fixed << ort_model::estimate_fps(path_model, paths_images, 640, 640, 3, true) << endl;

    return 0;
}
*/