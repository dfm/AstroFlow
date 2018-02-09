#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <limits>

#include "quad.h"

using namespace tensorflow;

REGISTER_OP("QuadLimbDarkeningRev")
  .Attr("T: {float, double}")
  .Input("u1: T")
  .Input("u2: T")
  .Input("p: T")
  .Input("z: T")
  .Input("bflux: T")
  .Output("bu1: T")
  .Output("bu2: T")
  .Output("bp: T")
  .Output("bz: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s, z;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &s));

    TF_RETURN_IF_ERROR(c->Merge(c->input(3), c->input(4), &z));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    c->set_output(2, c->input(2));
    c->set_output(3, c->input(3));
    return Status::OK();
  });

template <typename T>
class QuadLimbDarkeningRevOp : public OpKernel {
 public:
  explicit QuadLimbDarkeningRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& u1_tensor = context->input(0);
    const Tensor& u2_tensor = context->input(1);
    const Tensor& p_tensor = context->input(2);
    const Tensor& z_tensor = context->input(3);
    const Tensor& bflux_tensor = context->input(4);

    // Dimensions
    const int64 N = z_tensor.NumElements();
    OP_REQUIRES(context, (bflux_tensor.NumElements() == N), errors::InvalidArgument("Dimension mismatch"));

    // Output
    Tensor* bu1_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u1_tensor.shape(), &bu1_tensor));
    Tensor* bu2_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, u2_tensor.shape(), &bu2_tensor));
    Tensor* bp_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, p_tensor.shape(), &bp_tensor));
    Tensor* bz_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, z_tensor.shape(), &bz_tensor));

    // Access the data
    const auto u1 = u1_tensor.template scalar<T>()(0);
    const auto u2 = u2_tensor.template scalar<T>()(0);
    const auto p = p_tensor.template scalar<T>()(0);
    const auto z = z_tensor.template flat<T>();
    const auto bflux = bflux_tensor.template flat<T>();

    auto bu1 = bu1_tensor->template flat<T>();
    auto bu2 = bu2_tensor->template flat<T>();
    auto bp = bp_tensor->template flat<T>();
    auto bz = bz_tensor->template flat<T>();

    bu1(0) = 0.0;
    bu2(0) = 0.0;
    bp(0) = 0.0;

    typedef Eigen::Matrix<T, 4, 1> DerType;
    Eigen::AutoDiffScalar<DerType> ad_u1(u1, 4, 0),
                                   ad_u2(u2, 4, 1),
                                   ad_p(p, 4, 2);

    for (int64 n = 0; n < N; ++n) {
      Eigen::AutoDiffScalar<DerType> ad_z(z(n), 4, 3);
      auto f = astroflow::quad(ad_u1, ad_u2, ad_p, ad_z);
      auto d = f.derivatives();
      bu1(0) += bflux(n) * d(0);
      bu2(0) += bflux(n) * d(1);
      bp(0) += bflux(n) * d(2);
      bz(n) = bflux(n) * d(3);
    }
  }
};


#define REGISTER_KERNEL(type)                                                    \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("QuadLimbDarkeningRev").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      QuadLimbDarkeningRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
