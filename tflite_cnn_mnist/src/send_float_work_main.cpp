#include <Arduino.h>
#include <stdio.h>
#include <string.h>
#include "models/cnn_mnist_model.h"
#include "models/X_test_1_2.h"


#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
/*
// cnn_model * nn;
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
// TfLiteTensor *input;
// TfLiteTensor *output;
uint8_t *tensor_arena;
const int kArenaSize = 50000; 

float ** tensor_in = nullptr;
const int dim1 = 28;
const int dim2 = 28;
float serial_in[dim1*dim2];
// double i = 0.0;
// double serial_in[60];
double cmp = 1.001111111111111;

// double serial_in[dim1*dim2];
int counter1 = 0;
float* output;
const int ledPin = 5;

float ** array_2D_converter(float * arr_in, int dim1, int dim2) {
    int count = 0;
    float ** arr = new float*[dim1];
    for(int i = 0; i < dim1;i++){
        arr[i] = new float[dim2];
        for(int j = 0; j < dim2; j++) {
            arr[i][j] = arr_in[count];
            count++;
        }
    }
    printf("count %d\n\n",count);
    return(arr);
}


void setup()
{

  
  error_reporter = new tflite::MicroErrorReporter();
  model = tflite::GetModel(cnn_mnist_model_tflite);
    

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

  
  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(9600);
}

const int numBytes = 4;

void loop() {
  
  
  // put your main code here, to run repeatedly:
  byte array[numBytes];

  // spam the PC to indicate we're ready for data
  while (Serial.available() == 0)
  {
    //send '!' up the chain
    Serial.println('!');

    //spam the host until they respond :)
    delay(10);
  }

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
  serial_in[counter1] = value;
  counter1++;
  // print out received value
  Serial.print("Value on my system is: ");
  Serial.printf("%f\n",value);

  if(counter1 == (dim1*dim2) ) {

    
    for(int i = 0; i < counter1; i++) {
      Serial.printf("%lf,",serial_in[i]);
    }
    Serial.print("I am done here...go to 2d\n");
    

    float ** serial_in_2d = array_2D_converter(serial_in, dim1, dim2);
    

    int input = interpreter->inputs()[0];
    float* input_data_ptr = interpreter->typed_tensor<float>(input);
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
          *(input_data_ptr) = serial_in_2d[i][j];
          input_data_ptr++;
      }
    }

    int output_idx = interpreter->outputs()[0];
    interpreter->Invoke();
    output = interpreter->typed_tensor<float>(output_idx);
    Serial.print("  Tensorflow output ");
    for (int i = 0; i < 10; i++) {
      Serial.printf("%f, ", output[0], output[1]);
    }
    Serial.print("");

  }
  
  // int input = interpreter->inputs()[0];
  // float* input_data_ptr = interpreter->typed_tensor<float>(input);
  // for (int i = 0; i < dim1; i++) {
  //   for (int j = 0; j < dim2; j++) {
  //       *(input_data_ptr) = X_test_1_2[i][j];
  //       input_data_ptr++;
  //   }
  // }

  int output_idx = interpreter->outputs()[0];
  interpreter->Invoke();
  output = interpreter->typed_tensor<float>(output_idx);
  Serial.print("  Tensorflow output ");
  for (int i = 0; i < 10; i++) {
    Serial.printf("%f, ", output[i]);
  }
  Serial.println("");
  // delay so the light stays on
  delay(500);
  Serial.println("I am working");
  digitalWrite(LED_BUILTIN, LOW);

}

*/

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