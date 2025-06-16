#pragma once

#include <string>
#include <vector>
#include "cppflow/ops.h"

#ifdef _WIN64
#ifdef ML_EXPORTS
#define ML_API __declspec(dllexport)
#else
#define ML_API __declspec(dllimport)
#endif
#else
#define ML_API
#endif

namespace tfexec {
    int load_model(const std::string& model_path, unsigned int* model_id);
    int delete_model(unsigned int model_id);
    int predict(unsigned int model_id, 
        std::vector<std::tuple<std::string, cppflow::tensor>> data,
        std::vector<std::string> output_op_names,
        std::vector<cppflow::tensor> & output);
};

extern "C" {
    ML_API int tfexec__load_model(const char* model_path, int* model_id);
    ML_API int tfexec__register_output(int* model_id, const char* op_name);
    ML_API int tfexec__delete_model(int* model_id);

    ML_API int tfexec__provide_input(int* model_id, const char* op_name, int* num_shape_dims, int64_t* shape, float* data);
    ML_API int tfexec__predict(int* model_id);
    ML_API int tfexec__retrieve_output(int* model_id, const char* op_name, int* num_shape_dims, int64_t* shape, float* data);
    ML_API int tfexec__retrieve_output_no_check(int* model_id, const char* op_name, float* data);
};