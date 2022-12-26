#include <Arduino.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
// #include "cnn_model.h"
// #include "X_test_1_2.h"
#include "models/cnn_mnist_model.h"


#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"

// cnn_model * nn;

/*
namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} struct TfLiteTensor;

tflite::MicroMutableOpResolver<10> *resolver;
tflite::ErrorReporter *error_reporter;
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;
uint8_t *tensor_arena;
const int kArenaSize = 80000; 
float X_test_1_2_2[28][28] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,116,125,171,255,255,150,93,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,169,253,253,253,253,253,253,218,30,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,169,253,253,253,213,142,176,253,253,122,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,52,250,253,210,32,12,0,6,206,253,140,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,77,251,210,25,0,0,0,122,248,253,65,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,31,18,0,0,0,0,209,253,253,65,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,117,247,253,198,10,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,76,247,253,231,63,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,128,253,253,144,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,176,246,253,159,12,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,25,234,253,233,35,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,198,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,78,248,253,189,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,19,200,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,134,253,253,173,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,248,253,253,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,248,253,253,43,20,20,20,20,5,0,5,20,20,37,150,150,150,147,10,0},
    {0,0,0,0,0,0,0,0,248,253,253,253,253,253,253,253,168,143,166,253,253,253,253,253,253,253,123,0},
    {0,0,0,0,0,0,0,0,174,253,253,253,253,253,253,253,253,253,253,253,249,247,247,169,117,117,57,0},
    {0,0,0,0,0,0,0,0,0,118,123,123,123,166,253,253,253,155,123,123,41,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    };
void setup1(){
    // nn = new cnn_model();
    error_reporter = new tflite::MicroErrorReporter();
    model = tflite::GetModel(cnn_mnist_model_tflite);
    
    // model = tflite::GetModel(g_model);


    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddConv2D();
    resolver->AddMaxPool2D();
    resolver->AddSoftmax();

    tensor_arena = (uint8_t *)malloc(kArenaSize);

    if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }
    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize, error_reporter);
    
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }
    
    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    
    Serial.begin(115200);


}

void loop1(){
  float  (*number1)[28]= X_test_1_2_2;
  
  // I filled this vector, (dims are 28, 28)
  
  // std::vector<std::vector<int>> tensor;
  int input = interpreter->inputs()[0];
  float* input_data_ptr = interpreter->typed_tensor<float>(input);
  for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; j++) {
          *(input_data_ptr) = X_test_1_2_2[i][j];
          input_data_ptr++;
      }
  }
  // nn-> getInputBuffer()[0] =  number1;
  
  int output_idx = interpreter->outputs()[0];
  interpreter->Invoke();
  float* output = interpreter->typed_tensor<float>(output_idx);
  // std::cout << "OUTPUT: " << *output << std::endl;
  // input = number1;

  // const char *expected = number2 > number1 ? "True" : "False";

  // const char *predicted = result > 0.5 ? "True" : "False";

  // Serial.printf("Expected %d, Predicted %d\n", (int) (number1 * number1), (int) result);
  int i = 0;
  float tmp = 0.0;
  int idx = 0;
  for(i = 0;i < 10; i++) {

    // Change < to > if you want to find the smallest element
    if(tmp < output[i]) {
      tmp = 0.0;
      idx = i;
    }
    Serial.printf("output at array idx %d: %f\n",i,output[i]);
  }
  
  Serial.printf("Expected 2 | Actual 2\n");
  Serial.printf("Done\n\n");
  delay(1000);
  
}

*/