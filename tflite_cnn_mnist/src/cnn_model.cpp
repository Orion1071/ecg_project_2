#include "cnn_model.h"
#include "models/cnn_mnist_model.h"


#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"


/*

const int kArenaSize = 40000;
  
  
cnn_model::cnn_model()
{
    error_reporter = new tflite::MicroErrorReporter();
    model = tflite::GetModel(cnn_mnist_model_tflite);
    
    // model = tflite::GetModel(g_model);


    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddConv2D();
    resolver->AddMaxPool2D();

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
    input = interpreter->input(0);
    output = interpreter->output(0);
    // input = interpreter -> typed_input_tensor<unsigned int[28][28]>(0);

}

float *cnn_model::getInputBuffer()
{
    return input->data.f;
}

float cnn_model::predict()
{
    interpreter->Invoke();
    return output->data.f[0];
}

*/