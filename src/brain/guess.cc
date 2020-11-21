#include <cstdio>
#include <list>
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

//-------------------------------------------------------------------//

class NonCopyable {
protected:
    NonCopyable(NonCopyable const&) = delete;
    NonCopyable& operator=(NonCopyable const&) = delete;
    NonCopyable() = default;
};

//-------------------------------------------------------------------//

class StreamingStateItem : NonCopyable
{
  char  *i, *o, *r;
  size_t i_sz, o_sz, mv_sz;

public:
  StreamingStateItem(TfLiteTensor* i, TfLiteTensor* o)
  {
    TFLITE_MINIMAL_CHECK(o->type == kTfLiteFloat32);
    this->o = o->data.raw;
    o_sz = o->bytes;
    TFLITE_MINIMAL_CHECK(i->type == kTfLiteFloat32);
    this->i = i->data.raw;
    i_sz = i->bytes;
    TFLITE_MINIMAL_CHECK(i_sz > o_sz);
    TFLITE_MINIMAL_CHECK(i_sz % o_sz == 0);
    r = new char[i_sz]();
    TFLITE_MINIMAL_CHECK(r != nullptr);
    memset(this->i, 0, i_sz);
    mv_sz = i_sz - o_sz;
  }

  ~StreamingStateItem()
  {
    delete[] r;
  }

  void stream()
  {
    memmove(r, &r[o_sz], mv_sz);
    memcpy(&r[mv_sz], o, o_sz);
    memcpy(i, r, i_sz);
  }
};

//-------------------------------------------------------------------//

static void print_output(float* data, size_t sz_bytes)
{
  size_t len = sz_bytes / sizeof(*data);
  for(size_t i = 0, j = len - 1; i < len; i++)
  {
    printf("%f%c", data[i], i == j ? '\n' : ' ');
  }
}

//-------------------------------------------------------------------//

void guess(char* filename)
{
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  TFLITE_MINIMAL_CHECK(interpreter->inputs().size());

  auto input = interpreter->tensor(interpreter->inputs()[0]);
  TFLITE_MINIMAL_CHECK(input->type == kTfLiteFloat32);
  TFLITE_MINIMAL_CHECK(input->bytes % (sizeof(float) * 13) == 0);

  auto output = interpreter->tensor(interpreter->outputs()[0]);
  TFLITE_MINIMAL_CHECK(output->type == kTfLiteFloat32);
  TFLITE_MINIMAL_CHECK(output->bytes == 12 * sizeof(float));

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Streaming models have > 1 inputs outputs
  TFLITE_MINIMAL_CHECK(interpreter->inputs().size() == interpreter->outputs().size());
  std::list<StreamingStateItem> streaming_state;
  for(size_t i = 1; i < interpreter->inputs().size(); i++)
  {
    streaming_state.emplace_back(
      interpreter->tensor(interpreter->inputs()[i]),
      interpreter->tensor(interpreter->outputs()[i])
    );
  }

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
  print_output(output->data.f, output->bytes);

  // Does nothing for non-streaming model
  for(auto& item : streaming_state)
    item.stream();

  // Exit or continue
  start = scanf("%f", &input->data.f[0]);
  if(start == 1)
    goto loop;

  TFLITE_MINIMAL_CHECK(start == EOF);
}

//-------------------------------------------------------------------//

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "guess <tflite model>\n");
    return 1;
  }

  guess(argv[1]);

  return 0;
}
