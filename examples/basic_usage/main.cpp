#include <iostream>
#include <cassert>

#include "tfexec/tfexec.h"
#include "cppflow/tensor.h"

int main() {
    unsigned int model_id;
    tfexec::load_model("../model", &model_id);

    std::vector<int64_t> input_shape{10, 5};
    std::vector<float> input_data((std::vector<float>::size_type)50, 1);
    for (int i = 0; i < 50; i++) {
        input_data[i] = (float)i / 50.0;
    }

    std::vector<std::tuple<std::string, cppflow::tensor>> inputs;
    inputs.emplace_back("serving_default_input_1:0", cppflow::tensor(input_data, input_shape));

    // Expecting shape {10, 1} after running the model
    std::vector<std::string> output_ops{"StatefulPartitionedCall:0"};
    std::vector<cppflow::tensor> output;
    if (tfexec::predict(model_id, inputs, output_ops, output)) {
        std::cerr << "Error occured in predict." << std::endl;
        return 1;
    }

    if (output.size() != 1) {
        std::cerr << "Output had non-1 number of tensors: " << output.size() << std::endl;
        return 1;
    }

    auto out_shape = output.at(0).shape().get_data<int64_t>();
    auto out_data = output.at(0).get_data<float>();

    std::cout << "Output:" << std::endl;
    std::cout << "\tShape: (" << out_shape.at(0);
    for (size_t i = 1; i < out_shape.size(); i++)
        std::cout << ", " << out_shape.at(i);
    std::cout << ")" << std::endl;
    std::cout << "\tData:";
    for (size_t i = 0 ; i < out_data.size(); i++) {
        std::cout << " " << out_data.at(i);
    }
    std::cout << std::endl;

    assert(out_shape.size() == 2);
    assert(out_shape.at(0) == 10);
    assert(out_shape.at(1) == 1);
    assert(out_data.size() == 10);

    std::cout << "Example test passed!" << std::endl;

    return 0;
}
