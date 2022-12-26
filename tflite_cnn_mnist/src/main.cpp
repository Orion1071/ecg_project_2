#include <Arduino.h>
#include <stdio.h>
#include <string.h>
#include "models/cnn_mnist_model_2.h"
// #include "models/cnn_ecg_keras_small_4.h"
// #include "models/X_test_1_2.h"


#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"


namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} struct TfLiteTensor;

tflite::AllOpsResolver *resolver;
tflite::ErrorReporter *error_reporter;
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;
uint8_t *tensor_arena;
const int kArenaSize = 2500000; 

float ** tensor_in = nullptr;
const int dim1 = 28;
const int dim2 = 28;
float arr[dim1*dim2];
const int numBytes = 4;

// float* output;
const int ledPin = 5;

void array_2D_converter(float arr[dim1][dim2], float* arr_in, int dim1, int dim2) {
    int count = 0;
    // float ** arr = new float*[dim1];
    for (int i = 0; i < dim1; i++) {
        // arr[i] = new float[dim2];
        for (int j = 0; j < dim2; j++) {
            arr[i][j] = arr_in[count];
            count++;
        }
    }
    printf("count %d\n\n", count);
}


void setup()
{
  error_reporter = new tflite::MicroErrorReporter();
  model = tflite::GetModel(cnn_mnist_model_2_tflite);
  resolver = new tflite::AllOpsResolver();
  tensor_arena =  (uint8_t *)malloc(kArenaSize);
  if (!tensor_arena)
  {
      TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
      return;
  }
  // Build an interpreter to run the model with.
  interpreter = new tflite::MicroInterpreter(
      model, *resolver, tensor_arena, kArenaSize);
  // Allocate the tensor
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
  }
  size_t used_bytes = interpreter->arena_used_bytes();
  TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
}



void loop() {
  
  
    int curr_dim = 0;
    byte array[numBytes];

    while (Serial.available() == 0)
    {
      //send '!' up the chain
      Serial.println('!');

      //spam the host until they respond :)
      delay(10);
    }

    while (curr_dim < dim1 * dim2) {
      // turn on the LED to indicate we're waiting on data
      digitalWrite(LED_BUILTIN, HIGH);

      // wait until we have enough bytes
      while (Serial.available() < numBytes) {}

      for (int i = numBytes -1 ; i > -1; i--)
      {
        array[i] = Serial.read();
      }

      // print out what we received to just double check
      Serial.print("Byte array received was: 0x");
      for (int i = 0; i < numBytes; i++)
      {
        Serial.print(array[i], HEX);
      }
      Serial.println("");

          // now cast the 32 bits into something we want...
      float value = *((float*)(array));
      arr[curr_dim] = value;
      
      // print out received value
      Serial.print("Value on my system is: ");
      Serial.printf("%f\n",arr[curr_dim]);
      curr_dim ++;
      digitalWrite(LED_BUILTIN, LOW);

      Serial.println('$');
      delay(20);
    }

    Serial.print("The sum of the received array is: ");
    float mul = 0.0;
    for(int f = 0; f < dim1 * dim2; f++) {
      mul = mul + arr[f];
    }
    // Serial.printf("%f %f %f %f\n",arr[0],arr[1],arr[2],arr[3] );
    Serial.printf("%f\n", mul);


    // set up input data
    input = interpreter->input(0);
    for (int i = 0; i < dim1 * dim2; i++) {
          input->data.f[i] = arr[i];
    }
    // set up output data
    output = interpreter->output(0);
    // Run inference, and report any error.
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed\n");
    return;
    }
    Serial.print("  Tensorflow output ");
    float max = 0.0;
    int index = -1;
    // Iterate the array
    for(int i=0;i<10;i++)
    {
        if(output->data.f[i]>max)
        {
            // If current value is greater than max
            // value then replace it with max value
            max = output->data.f[i];
            index = i;
        }
    }
    Serial.printf("Value: %f Prediction: %d\n", max, index);
    // delay so the light stays on
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("%");
    curr_dim = 0;

}


// float f;
// //get b1,b2,b3 from serial.read()
// char b0= Serial.read();
// char b1= Serial.read();
// char b2= Serial.read();
// char b3= Serial.read();
// char b[] = {b3, b2, b1, b0};
// memcpy(&f, &b, sizeof(f));
// return f;



/*
  // int input = interpreter->inputs()[0];
  // float* input_data_ptr = interpreter->typed_tensor<float>(input);
  // for (int i = 0; i < dim1; i++) {
  //   for (int j = 0; j < dim2; j++) {
  //       *(input_data_ptr) = X_test_1_2[i][j];
  //       input_data_ptr++;
  //   }
  // }

  // int output_idx = interpreter->outputs()[0];
  // interpreter->Invoke();
  // output = interpreter->typed_tensor<float>(output_idx);
  // Serial.print("  Tensorflow output ");
  // for (int i = 0; i < 10; i++) {
  //   Serial.printf("%f, ", output[i]);
  // }
  // Serial.println("");
set up
---------------------------
// nn = new cnn_model();
    error_reporter = new tflite::MicroErrorReporter();
    model = tflite::GetModel(cnn_ptb_model_2_tflite);
    

    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddConv2D();
    resolver->AddMaxPool2D();
    resolver->AddSoftmax();
    resolver->AddRelu();
    // resolver->Add
    
    

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

    -------------------------------------

    loop 

    ------------------------------
    if (arr != NULL) {
        float*in = (float*) arr;
        int input = interpreter->inputs()[0];
        float* input_data_ptr = interpreter->typed_tensor<float>(input);
        for (int j = 0; j < 100; j++) {
            *(input_data_ptr) = arr[j];
        }

        int output_idx = interpreter->outputs()[0];
        interpreter->Invoke();
        output = interpreter->typed_tensor<float>(output_idx);
        Serial.printf("%f %f\n", output[0], output[1]);
        arr = NULL;
    }


*/