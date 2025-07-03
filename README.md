# TF Exec

This library gives an easy interface for loading and executing deep learning models written in Tensorflow.

## Prerequisites
- Cmake >= 3.10
- Windows: Visual Studio
- Linux: g/g++ compiler
- Tensorflow installation: https://www.tensorflow.org/install/gpu
- Install the Tensorflow C-Language Bindings: https://www.tensorflow.org/install/lang_c. If doing this on Windows, modify the windows PATH to include the directory. If this is not possible copy the contents of the `lib` directory to your build directory to appropriately link it. The library version may need to be edited in the CMakeLists.txt file depending on which Tensorflow version was used to train your model.

## Building

### Windows
1. Make a `build` directory in the project's root directory.
2. Open CMake and specify the build directory and source directories.
3. Click configure, choosing Visual Studio Project
4. Generate and Open Project
5. Build in Visual Studio

### Linux
1. Make a `build` directory in the project's root directory and navigate to it in a shell.
2. Run cmake within the `build` directory, referencing the parent directory. `cmake ..`
3. Run `make`

### Building Examples
The instructions for building examples are the same as building the library, however you change the source directory to the example and make the `build` directory in the example directory.

## API

The API is split into two sections. Firstly, the underlying C++ api and secondly the Fortran API built in the fortran example.

### C++

The underlying source of the project uses the C++17 standard and is (hopefully) intuititive to use. Each function returns 0 if it completed successfully and 1 if there was an error. The C++ API contains three functions:

- `int load_model(const std::string& model_path, unsigned int* model_id)`: Load a model into memory. Provide a path to the model (relative to the working directory or absolute) and the address to a uint that will be filled with the id that references the model when calling to other functions.
- `int delete_model(unsigned int model_id)`: Delete a model that is no longer needed. It is not necessary to call this when cleaning up, as models are automatically deallocated by the default destructor of their container.
- `int predict(unsigned int model_id, std::vector<std::tuple<std::string, cppflow::tensor>> data, std::vector<std::string> output_op_names, std::vector<cppflow::tensor> & output)`: Run a model forward with specific data. Input is formatted as a vector of pairs of operation names that each map to a tensor. Output operation names should also be given and a reference to a vector of tensors to be filled. The output will be filled in the order provided by `output_op_names`.

### Fortran

The Fortran bindings to the C/C++ library can be found in `examples/fortran/src/ML_modules.f90`. The functions provided are explained in detail below:

- `ML_load_model(model_path, model_id)`: Loads a model from directory `model_path` into memory and stores the associated id into `model_id`.
- `ML_register_output(model_id, op_name)`: Register an operation name as an output of the model. When calling predict, only registered output operations will be stored for later retrieval.
- `ML_delete_model(model_id)`: Delete the associated model from memory.
- `ML_provide_input(model_id, op_name, shape, data_in)`: Provides an input to the model to be processed when the predict function is called. The `shape` parameter must be an array of `INTEGER(kind=C_INT64_T)` and `data_in` must be a flattened array of type `REAL(kind=C_FLOAT)` with a number of elements matching the product of the shape elements.
- `ML_predict(model_id)`: Run the model with given inputs, recording registered outputs for later retrieval.
- `ML_retrieve_output(model_id, op_name, shape, data_out)`: Retrieves data after running predict. The `shape` parameter in this function is to check that the expected shape is equal to what was actually generated and must be an array of type `INTEGER(kind=C_INT64_T)`. The `data_out` parameter should be a flattened array of the appropriate size, again of type `REAL(kind=C_FLOAT)`.
- `ML_retrieve_output_no_check(model_id, op_name, data_out)`: This function operates in the same way as `ML_retrieve_output`, however it does not check the size of the tensor, which may lead to minimal gains in performance.

Note that while numerical values passed into these functions must be `C_`-types, strings may be passed in as `CHARACTER*(*)` as each is converted in the interface functions.

## Finding Operation Names

Tensorflow comes with a convenient CLI tool for disecting saved models. `saved_model_cli` is loaded into the path when tensorflow is installed. If your model is saved to directory `x/y/z`, then you can view the input and output operation names by running: `saved_model_cli show --dir x/y/z --tag serve --signature_def serving_default`. This will produce output that looks like:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['args_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 5)
      name: serving_default_args_0:0
  inputs['args_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 4)
      name: serving_default_args_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall:0
  outputs['output_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 2)
      name: StatefulPartitionedCall:1
Method name is: tensorflow/serving/predict
```
From here, we can see that the input operation names are `serving_default_args_0:0` and `serving_default_args_1:0` while the output operation names are `StatefulPartitionedCall:0` and `StatefulPartitionedCall:1`. All input operation names must be provided an input when calling predict. However, not all operation outputs need to be collected or registered.