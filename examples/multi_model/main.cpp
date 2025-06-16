#include <iostream>
#include <cassert>

#include "tfexec/tfexec.h"

int run_model(unsigned int model_id, size_t output_channels) {
    std::cerr << "Running model " << model_id << std::endl;
    std::vector<int64_t> shape{10, 10};
    std::vector<float> data((std::vector<float>::size_type)100, 1);
    for (int i = 0; i < data.size(); i++) {
        data[i] = (float)i / data.size();
    }

    cppflow::tensor input_tensor(data, shape);
    std::vector<std::tuple<std::string, cppflow::tensor>> inputs{ {"serving_default_args_0:0", input_tensor} };

    std::vector<std::string> output_op_names{ "StatefulPartitionedCall:0" };
    std::vector<cppflow::tensor> outputs;
    
    if (tfexec::predict(model_id, inputs, output_op_names, outputs)) {
        std::cerr << "Error occured in predict for model " << std::endl;
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

    assert(outputs.size() == 1);
    assert(outputs.at(0).shape().get_data<int64_t>().size() == 2);
    assert(outputs.at(0).shape().get_data<int64_t>().at(0) == 10);
    assert(outputs.at(0).shape().get_data<int64_t>().at(1) == output_channels);

    return 0;
}

int main() {
    unsigned int model_1_id, model_2_id;
    if (tfexec::load_model("../models/model_1", &model_1_id)) {
        std::cerr << "Error loading model 1" <<std::endl;
        return 1;
    }
    if (tfexec::load_model("../models/model_2", &model_2_id)) {
        std::cerr << "Error loading model 2" << std::endl;
        return 1;
    }

    if (run_model(model_1_id, 1))
        return 1;
    if (run_model(model_2_id, 2))
        return 1;

    std::cout << "All examples passed!" << std::endl;
    return 0;
}
