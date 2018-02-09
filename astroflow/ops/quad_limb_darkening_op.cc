#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <limits>

#include "quad.h"

using namespace tensorflow;

REGISTER_OP("QuadLimbDarkening")
  .Attr("T: {float, double}")
  .Input("u1: T")
  .Input("u2: T")
  .Input("p: T")
  .Input("z: T")
  .Output("flux: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &s));
    c->set_output(0, c->input(3));
    return Status::OK();
  });

template <typename T>
class QuadLimbDarkeningOp : public OpKernel {
 public:
  explicit QuadLimbDarkeningOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& u1_tensor = context->input(0);
    const Tensor& u2_tensor = context->input(1);
    const Tensor& p_tensor = context->input(2);
    const Tensor& z_tensor = context->input(3);

    // Dimensions
    const int64 N = z_tensor.NumElements();

    // Output
    Tensor* flux_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &flux_tensor));

    // Access the data
    const auto u1 = u1_tensor.template scalar<T>()(0);
    const auto u2 = u2_tensor.template scalar<T>()(0);
    const auto p = p_tensor.template scalar<T>()(0);
    const auto z = z_tensor.template flat<T>();
    auto flux = flux_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      flux(n) = astroflow::quad<T>(u1, u2, p, z(n));
    }
  }
};


#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("QuadLimbDarkening").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      QuadLimbDarkeningOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
