#include <iostream>
#include <cassert>

#include "tfexec/tfexec.h"
#include "cppflow/tensor.h"

int main() {
    unsigned int model_id;
    tfexec::load_model("../model", &model_id);

    std::vector<float> input_data_1{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 };
    std::vector<float> input_data_2{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };
    std::vector<int64_t> input_shape_1{3, 5};
    std::vector<int64_t> input_shape_2{3, 4};

    cppflow::tensor input_tensor_1(input_data_1, input_shape_1);
    cppflow::tensor input_tensor_2(input_data_2, input_shape_2);

    std::vector<std::tuple<std::string, cppflow::tensor>> inputs;
    inputs.emplace_back("serving_default_args_0:0", input_tensor_1);
    inputs.emplace_back("serving_default_args_1:0", input_tensor_2);

    std::vector<std::string> output_op_names{ "StatefulPartitionedCall:0", "StatefulPartitionedCall:1" };
    std::vector<cppflow::tensor> outputs;
    
    if (tfexec::predict(model_id, inputs, output_op_names, outputs)) {
        std::cerr << "Error occured in predict." << std::endl;
        return 1;
    }

    if (outputs.size() != output_op_names.size()) {
        std::cerr << "Error: expected " << output_op_names.size() << " outputs, but received " << outputs.size() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        auto & output_tensor = outputs.at(i);
        std::cout << "Output:" << output_op_names.at(i) << std::endl;
        
        auto out_shape = output_tensor.shape().get_data<int64_t>();
        std::cout << "\tShape: (" << out_shape.at(0);
        for (size_t j = 1; j < out_shape.size(); j++)
            std::cout << ", " << out_shape.at(j);
        std::cout << ")" << std::endl;

        auto output = output_tensor.get_data<float>();
        std::cout << "\tData:";
        for (size_t j = 0; j < output.size(); j++)
            std::cout << " " << output.at(j);
        std::cout << std::endl << std::endl;
    }

    assert(outputs.size() == 2);
    assert(outputs.at(0).shape().get_data<int64_t>().size() == 2);
    assert(outputs.at(1).shape().get_data<int64_t>().size() == 2);
    assert(outputs.at(0).shape().get_data<int64_t>().at(0) == 3);
    assert(outputs.at(0).shape().get_data<int64_t>().at(1) == 1);
    assert(outputs.at(1).shape().get_data<int64_t>().at(0) == 3);
    assert(outputs.at(1).shape().get_data<int64_t>().at(1) == 2);
    
    std::cout << "All asserts passed!" << std::endl;
    tfexec::delete_model(model_id);

    return 0;
}
