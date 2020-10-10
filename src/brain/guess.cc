#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

static_assert(sizeof(float) == 4, "WTF");

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "guess <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  TFLITE_MINIMAL_CHECK(interpreter->inputs().size() == 1);
  auto input = interpreter->tensor(interpreter->inputs()[0]);
  TFLITE_MINIMAL_CHECK(input->type == kTfLiteFloat32);
  TFLITE_MINIMAL_CHECK(input->bytes % (sizeof(float) * 13) == 0);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  int start = 0;

loop:

  // Fill input
  for(size_t i = start; i < input->bytes / sizeof(float); i++)
  {
    TFLITE_MINIMAL_CHECK(scanf("%f", &input->data.f[i]) == 1);
  }
  TFLITE_MINIMAL_CHECK(getchar() == '\n');

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Result
  TFLITE_MINIMAL_CHECK(interpreter->outputs().size() == 1);
  auto output = interpreter->tensor(interpreter->outputs()[0]);
  TFLITE_MINIMAL_CHECK(output->type == kTfLiteFloat32);
  TFLITE_MINIMAL_CHECK(output->bytes == 12 * sizeof(float));
  for(size_t i = 0; i < output->bytes / sizeof(float); i++)
  {
    printf("%f%c", output->data.f[i], i == output->bytes / sizeof(float) - 1 ? '\n' : ' ');
  }

  start = scanf("%f", &input->data.f[0]);
  if(start == 1)
    goto loop;

  TFLITE_MINIMAL_CHECK(start == EOF);

  return 0;
}
