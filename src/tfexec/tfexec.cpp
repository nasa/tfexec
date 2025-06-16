#include "tfexec/tfexec.h"

#include <iostream>
#include <memory>
#include <map>
#include <list>

#include "cppflow/ops.h"
#include "cppflow/model.h"

// C++ Structures
static std::list<std::unique_ptr<cppflow::model>> models;
static std::map<unsigned int, std::list<std::unique_ptr<cppflow::model>>::iterator> model_map;
static unsigned int largest_id = 0;

// C Interface Structures
static std::map<unsigned int, std::vector<std::string>> model_outputs;
static std::map<unsigned int, std::vector<std::tuple<std::string, cppflow::tensor>>> input_placeholder;
static std::map<unsigned int, std::map<std::string, cppflow::tensor>> output_placeholder;


int tfexec::load_model(const std::string& model_path, unsigned int* model_id) {
    *model_id = largest_id++;
    models.push_back(std::make_unique<cppflow::model>(model_path));
    model_map[*model_id] = models.end();
    model_map[*model_id]--;
    return 0;
}

int tfexec::delete_model(unsigned int model_id) {
    models.erase(model_map.at(model_id));
    model_map.erase(model_map.find(model_id));
    return 0;
}

int tfexec::predict(unsigned int model_id, 
        std::vector<std::tuple<std::string, cppflow::tensor>> data_in,
        std::vector<std::string> output_op_names,
        std::vector<cppflow::tensor> & output) {
    output = model_map.at(model_id)->get()->operator()(data_in, output_op_names);
    return 0;
}

extern "C" {
    int tfexec__load_model(const char* model_path, int* model_id) {
        try {
            auto result = tfexec::load_model(std::string(model_path), (unsigned int*)model_id);
            unsigned int model_id_uint = *model_id;
            model_outputs[model_id_uint] = {};
            input_placeholder[model_id_uint] = {};
            output_placeholder[model_id_uint] = {};
            return result;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__load_model: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__register_output(int* model_id, const char* output_op_name) {
        try {
            model_outputs.at((unsigned int)(*model_id)).push_back(std::string(output_op_name));
            return 0;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__register_output: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__delete_model(int* model_id) {
        try {
            return tfexec::delete_model(*model_id);
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__delete_model: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__provide_input(int* model_id, const char* op_name, int* num_shape_dims, int64_t* shape, float* data) {
        try {
            unsigned int model_id_uint = *model_id;
            std::vector<int64_t> shape_vec(shape, shape + *num_shape_dims);

            unsigned int num_elements = 1;
            for (auto dim : shape_vec)
                num_elements *= dim;

            std::vector<float> data_vec(data, data + num_elements);
            cppflow::tensor input_tensor(data_vec, shape_vec);

            input_placeholder.at(model_id_uint).emplace_back(op_name, input_tensor);
            return 0;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__provide_input: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__predict(int* model_id) {
        try {
            unsigned int model_id_uint = *model_id;
            output_placeholder.at(model_id_uint).clear();
            auto & op_names = model_outputs.at(model_id_uint);
            std::vector<cppflow::tensor> outputs;

            auto result = tfexec::predict(model_id_uint, input_placeholder.at(model_id_uint), op_names, outputs);

            for (size_t i = 0; i < outputs.size(); i++) {
                output_placeholder.at(model_id_uint)[op_names.at(i)] = std::move(outputs.at(i));
            }

            input_placeholder.at(model_id_uint).clear();
            return result;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__predict: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__retrieve_output(int* model_id, const char* op_name, int* num_shape_dims, int64_t* shape, float* data) {
        try {
            // Retrieve output given name and model id
            unsigned int model_id_uint = *model_id;
            std::string op_name_str(op_name);
            cppflow::tensor & output = output_placeholder.at(model_id_uint).at(op_name);
            
            // Check output shape matches expectations
            std::vector<int64_t> calc_shape = output.shape().get_data<int64_t>();
            bool shape_error = false;
            if (calc_shape.size() != (std::vector<int64_t>::size_type)(*num_shape_dims)) {
                shape_error = true;
            }
            if (!shape_error) {
                for (size_t i = 0; i < calc_shape.size(); i++) {
                    if (calc_shape.at(i) != shape[i]) {
                        shape_error = true;
                        break;
                    }
                }
            }
            if (shape_error) {
                std::cerr << "Error: For output " << op_name_str << "expected tensor with shape (" << shape[0];
                for (size_t i = 1; i < *num_shape_dims; i++)
                    std::cerr << ", " << shape[i];
                std::cerr << "), but received tensor with shape (" << calc_shape.at(0);
                for (size_t i = 1; i < calc_shape.size(); i++)
                    std::cerr << ", " << calc_shape.at(i);
                std::cerr << ")" << std::endl;
                return 1;
            }

            // Copy tensor into pre-allocated buffer
            size_t num_dims = 1;
            for (auto dim : calc_shape)
                num_dims *= dim;
            memcpy(data, output.get_data<float>().data(), sizeof(float) * num_dims);

            return 0;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__retrieve_output: " << error.what() << std::endl;
            return 1;
        }
    }

    int tfexec__retrieve_output_no_check(int* model_id, const char* op_name, float* data) {
        try {
            unsigned int model_id_uint = *model_id;
            std::string op_name_str(op_name);
            cppflow::tensor & output = output_placeholder.at(model_id_uint).at(op_name_str);

            std::vector<int64_t> calc_shape = output.shape().get_data<int64_t>();
            size_t num_dims = 1;
            for (auto dim : calc_shape)
                num_dims *= dim;
            memcpy(data, output.get_data<float>().data(), sizeof(float) * num_dims);
            return 0;
        } catch (const std::runtime_error& error) {
            std::cerr << "TFExec encountered a runtime error in tfexec__retrieve_output_no_check: " << error.what() << std::endl;
            return 1;
        }
    }
};