#include <jni.h>
#include <iostream>
#include <vector>
#include <tflite.h>
#include <torch.h>
#include <ort.h>

using namespace std;

void split_string_by_symbol(char *str, char symbol, vector<string> &splits) {
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
    if (!buffer.empty()) {
        splits.push_back(buffer);
    }
}

extern "C" JNIEXPORT jdouble JNICALL Java_com_example_nativespeedapp_MainActivity_estimate_1tflite_1fps(
        JNIEnv* env, jobject, jstring path_model, jstring paths_images,
        jint height, jint width, jint n_channels, jboolean torch_input_mode) {

    char*           converted_path_model;
    char*           converted_paths_images;
    vector<string>  paths_images_vec;
    double          fps;

    converted_path_model = (char *)env->GetStringUTFChars( path_model, nullptr );
    converted_paths_images = (char *)env->GetStringUTFChars( paths_images, nullptr );
    split_string_by_symbol(converted_paths_images, ';', paths_images_vec);

    fps = tflite_model::estimate_fps(converted_path_model, paths_images_vec,
                                    height, width, n_channels, torch_input_mode);
    return fps;
}

extern "C" JNIEXPORT jdouble JNICALL Java_com_example_nativespeedapp_MainActivity_estimate_1torch_1fps(
        JNIEnv* env, jobject, jstring path_model, jstring paths_images,
        jint height, jint width, jint n_channels) {

    char*           converted_path_model;
    char*           converted_paths_images;
    vector<string>  paths_images_vec;
    double          fps;

    converted_path_model = (char *)env->GetStringUTFChars( path_model, nullptr );
    converted_paths_images = (char *)env->GetStringUTFChars( paths_images, nullptr );
    split_string_by_symbol(converted_paths_images, ';', paths_images_vec);

    fps = torch_model::estimate_fps(converted_path_model, paths_images_vec,
                                     height, width, n_channels);
    return fps;
}

extern "C" JNIEXPORT jdouble JNICALL Java_com_example_nativespeedapp_MainActivity_estimate_1ort_1fps(
        JNIEnv* env, jobject, jstring path_model, jstring paths_images,
        jint height, jint width, jint n_channels, jboolean torch_input_mode) {

    char*           converted_path_model;
    char*           converted_paths_images;
    vector<string>  paths_images_vec;
    double          fps;

    converted_path_model = (char *)env->GetStringUTFChars( path_model, nullptr );
    converted_paths_images = (char *)env->GetStringUTFChars( paths_images, nullptr );
    split_string_by_symbol(converted_paths_images, ';', paths_images_vec);

    fps = ort_model::estimate_fps(converted_path_model, paths_images_vec,
                                  height, width, n_channels, torch_input_mode);
    return fps;
}