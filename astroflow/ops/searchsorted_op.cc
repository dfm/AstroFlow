#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Searchsorted")
  .Attr("T: {float, double}")
  .Attr("check_sorted: bool = true")
  .Input("a: T")
  .Input("v: T")
  .Output("inds: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle a, v;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v));
    c->set_output(0, c->input(1));
    return Status::OK();
  });

template <typename T>
class SearchsortedOp : public OpKernel {
 public:
  explicit SearchsortedOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("check_sorted", &check_sorted_));
  }

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& a_tensor = context->input(0);
    const Tensor& v_tensor = context->input(1);

    OP_REQUIRES(context, (a_tensor.dims() == 1), errors::InvalidArgument("'a' should be 1-dimensional"));
    OP_REQUIRES(context, ((v_tensor.dims() == 0) || (v_tensor.dims() == 1)), errors::InvalidArgument("'v' should be a scalar or 1-dimensional"));

    // Dimensions
    int64 m = 0;
    const int64 N = a_tensor.NumElements();
    const int64 M = v_tensor.NumElements();

    // Access the data
    const auto a = a_tensor.template flat<T>();
    const auto v = v_tensor.template flat<T>();

    // Check for sorted order
    if (check_sorted_) {
      for (int64 n = 0; n < N - 1; ++n)
        OP_REQUIRES(context, (a(n+1) > a(n)), errors::InvalidArgument("'a' must be sorted"));
      for (int64 m = 0; m < M - 1; ++m)
        OP_REQUIRES(context, (v(m+1) > v(m)), errors::InvalidArgument("'v' must be sorted"));
    }

    // Output
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_tensor.shape(), &inds_tensor));
    auto inds = inds_tensor->flat<int64>();

    while ((m < M) && (v(m) <= a(0))) {
      inds(m) = 0;
      m++;
    }
    if (m >= M) return;

    for (int64 n = 0; n < N-1; ++n) {
      while (v(m) <= a(n+1)) {
        inds(m) = n+1;
        m++;
        if (m >= M) return;
      }
    }

    while (m < M) {
      inds(m) = N;
      m++;
    }
  }
 private:
  bool check_sorted_;
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Searchsorted").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SearchsortedOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
