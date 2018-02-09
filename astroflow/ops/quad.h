#ifndef _ASTROFLOW_OPS_ELLINT_H_
#define _ASTROFLOW_OPS_ELLINT_H_

//
// Elliptic integrals computed following:
//  Bulirsch 1965, Numerische Mathematik, 7, 78
//  Bulirsch 1965, Numerische Mathematik, 7, 353
//
// And the implementation by E. Agol (private communication).
//

#include <cmath>

#include "AutoDiffScalar.h"

namespace astroflow {

#define ELLINT_CONV_TOL 1.0e-8
#define ELLINT_MAX_ITER 200

  // K: 1.0 - k^2 >= 0.0
  template <typename T>
  T ellint_1 (const T& k) {
    T kc = std::sqrt(1.0 - k * k), m = T(1.0), h;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      h = m;
      m += kc;
      if (std::abs(h - kc) / h <= ELLINT_CONV_TOL) break;
      kc = sqrt(h * kc);
      m *= 0.5;
    }
    return M_PI / m;
  }

  // E: 1.0 - k^2 >= 0.0
  template <typename T>
  T ellint_2 (const T& k) {
    T b = 1.0 - k * k, kc = std::sqrt(b), m = T(1.0), c = T(1.0), a = b + 1.0, m0;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      b = 2.0 * (c * kc + b);
      c = a;
      m0 = m;
      m += kc;
      a += b / m;
      if (std::abs(m0 - kc) / m0 <= ELLINT_CONV_TOL) break;
      kc = 2.0 * std::sqrt(kc * m0);
    }
    return M_PI_4 * a / m;
  }

  // Pi: 1.0 - k^2 >= 0.0 & 0.0 <= n < 1.0 (doesn't seem consistent for n < 0.0)
  template <typename T>
  T ellint_3 (const T& k, const T& n) {
    T kc = std::sqrt(1.0 - k * k), p = std::sqrt(1.0 - n), m0 = 1.0, c = 1.0, d = 1.0 / p, e = kc, f, g;
    for (int i = 0; i < ELLINT_MAX_ITER; ++i) {
      f = c;
      c += d / p;
      g = e / p;
      d = 2.0 * (f * g + d);
      p = g + p;
      g = m0;
      m0 = kc + m0;
      if (std::abs(1.0 - kc / g) <= ELLINT_CONV_TOL) break;
      kc = 2.0 * sqrt(e);
      e = kc * m0;
    }
    return M_PI_2 * (c * m0 + d) / (m0 * (m0 + p));
  }

#undef ELLINT_CONV_TOL
#undef ELLINT_MAX_ITER

  // Gradients.
  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_1 (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar value = z.value(),
              Kz = ellint_1(value),
              Ez = ellint_2(value),
              z2 = value * value;
    return Eigen::AutoDiffScalar<T>(
      Kz,
      z.derivatives() * (Ez / (1.0 - z2) - Kz) / value
    );
  }

  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_2 (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar value = z.value(),
              Kz = ellint_1(value),
              Ez = ellint_2(value);
    return Eigen::AutoDiffScalar<T>(
      Ez,
      z.derivatives() * (Ez - Kz) / value
    );
  }

  template <typename T>
  Eigen::AutoDiffScalar<T> ellint_3 (const Eigen::AutoDiffScalar<T>& k,
                                    const Eigen::AutoDiffScalar<T>& n)
  {
    typename T::Scalar k_value = k.value(),
              n_value = n.value(),
              Kk = ellint_1(k_value),
              Ek = ellint_2(k_value),
              Pnk = ellint_3(k_value, n_value),
              k2 = k_value * k_value,
              n2 = n_value * n_value;
    return Eigen::AutoDiffScalar<T>(
      Pnk,
      (n.derivatives() * 0.5*(Ek + (Kk*(k2-n_value) + Pnk*(n2-k2))/n_value) / (n_value-1.0) -
      k.derivatives() * k_value * (Ek / (k2 - 1.0) + Pnk)) / (k2-n_value)
    );
  }

// Constants from Table A1.
#define PAL_AB  T a = (p-z)*(p-z), \
                  b = (p+z)*(p+z);

#define PAL_K   T k0=std::acos((p2+z2-1.0)/(2.0*p*z)), \
                  k1=std::acos((1.0+z2-p2)/(2.0*z));

#define PAL_CI  T ci = 2.0/(9.0*M_PI*std::sqrt(1.0-a)), \
                  cik = (1.0-5.0*z2+p2+a*b), \
                  cie = (z2+7.0*p2-4.0)*(1.0-a), \
                  cip = -3.0*(p+z)/(p-z); \
                fk = ci*cik; fe = ci*cie; fp = ci*cip; \
                kk = ee = pp = 1; \
                k = std::sqrt(4.0*z*p/(1.0-a)); \
                n = (a-b)/a; \
                f0 = p2; \
                f2 = 0.5*f0*(f0+2.0*z2);

#define PAL_CG  PAL_K \
                T cg = 1.0/(9.0*M_PI*std::sqrt(p*z)), \
                  cgk = ((1.0-b)*(2.0*b+a-3.0)-3.0*(p+z)*(p-z)*(b-2.0)), \
                  cge = 4.0*p*z*(z2+7.0*p2-4.0), \
                  cgp = -3.0*(p+z)/(p-z); \
                fk = cg*cgk; fe = cg*cge; fp = cg*cgp; \
                kk = ee = pp = 1; \
                k = std::sqrt((1.0-a)/(4.0*p*z)); \
                n = (a-1.0)/a; \
                f0 = (p2*k0+k1-std::sqrt(z2-0.25*(1.0+z2-p2)*(1.0+z2-p2)))/M_PI; \
                f2 = (k1+p2*(p2+2.0*z2)*k0-0.25*(1.0+5.0*p2+z2)*std::sqrt((1.0-a)*(b-1.0)))/(2.0*M_PI);

  template <typename T>
  T quad (const T& u1, const T& u2, const T& p, const T& z0)
  {
    int kk = 0, ee = 0, pp = 0;
    T eps = std::numeric_limits<T>::epsilon();

    T z = std::abs(z0);
    if (z + p >= 1.0) return T(1.0);

    T w0, w1, w2,
      f0 = T(0.0), f1 = T(0.0), fk = T(0.0), fe = T(0.0), fp = T(0.0),
      f2 = T(0.0), n = T(0.0), k = T(0.0),
      df, z2, p2;

    z2 = z * z;
    p2 = p * p;

    // Pre-compute the constant coefficients.
    w0 = (6.0-6.0*u1-12.0*u2) / (6.0-2.0*u1-u2);
    w1 = ( 6.0*u1+12.0*u2   ) / (6.0-2.0*u1-u2);
    w2 = (   6.0*u2         ) / (6.0-2.0*u1-u2);

    // Run through the cases and compute the coefficients from Table A1&2.
    if (z <= 0.0 && p <= 1.0) {
      // M&A 10, Pal A
      f0 = p2;
      f1 = 2.0/3.0*(1.0-(1.0-f0)*sqrt(1.0-f0));
      f2 = 0.5*f0*f0;
    } else if (z <= p-1.0) {
      // M&A 11, Pal A_G
      f0 = T(1.0);
      f1 = T(2.0/3.0);
      f2 = T(0.5);
    } else if (z < p && z < 1.0 - p - eps) {
      // M&A 9, Pal B
      PAL_AB PAL_CI
        f1 = T(2.0/3.0);
    } else if (z < p && std::abs(z-1.0+p) <= eps) {
      // M&A -, Pal B_T
      f0 = p2;
      f1 = (2.0/(3.0*M_PI)*acos(1.0-2.0*p)-4.0/(9.0*M_PI)
          * (3.0+2.0*p-8.0*f0)*sqrt(p*(1.0-p)));
      f2 = 0.5*f0*(f0+2.0*z2);
    } else if (z < p) {
      // M&A 8, Pal B_G
      PAL_AB PAL_CG
        f1 = T(2.0/3.0);
    } else if (std::abs(z-p) <= eps && z < 1.0-p-eps) {
      // M&A 5, Pal C
      T t = T(2.0/(9.0*M_PI));
      f0 = p2;
      f1 = T(1.0/3.0);
      fk = t*(1.0-4.0*f0); kk = 1;
      fe = t*4.0*(2.0*f0-1.0); ee = 1;
      f2 = 1.5*f0*f0;
      k = 2.0*p;
    } else if (std::abs(z-p) <= eps && std::abs(z-1.0+p) <= eps) {
      // M&A 6, Pal C_T
      f0 = T(0.25);
      f1 = T(1.0/3.0-4.0/(9.0*M_PI));
      f2 = T(3.0/32.0);
    } else if (std::abs(z-p) <= eps) {
      // M&A 7, Pal C_G
      PAL_AB PAL_K
        f0 = (p2*k0+k1-sqrt(z2-0.25*(1.0+z2-p2)*(1.0+z2-p2)))/M_PI;
      f1 = T(1.0/3.0);
      fk = -(1.0-4.0*p2)*(3.0-8.0*p2)/(9.0*M_PI*p); kk = 1;
      fe = 16.0*p*(2.0*p2-1.0)/(9.0*M_PI); ee = 1;
      f2 = (k1+p2*(p2+2.0*z2)*k0-0.25*(1.0+5.0*p2+z2)*sqrt((1.0-a)*(b-1.0)))/(2.0*M_PI);
      k = 1.0/(2.0*p);
    } else if (z < 1.0-p-eps) {
      // M&A 3, Pal D
      PAL_AB PAL_CI
    } else if (std::abs(z-1.0+p) <= eps) {
      // M&A 4, Pal E
      f0 = p2;
      f1 = (2.0/(3.0*M_PI)*acos(1.0-2.0*p)-4.0/(9.0*M_PI)
          *(3.0+2.0*p-8.0*f0)*sqrt(p*(1.0-p)));
      f2 = 0.5*f0*(f0+2.0*z2);
    } else if (z < 1.0+p-eps) {
      // M&A 2, Pal F
      PAL_AB PAL_CG
    }

    // Compute the base flux change.
    df = w0*f0+w1*f1+w2*f2;

    // Add in the elliptic integral terms.
    if (kk && fk != 0.0)
      df += w1 * fk * ellint_1(k);
    if (ee && fe != 0.0)
      df += w1 * fe * ellint_2(k);
    if (pp && fp != 0.0)
      df += w1 * fp * ellint_3(-k, n);

    return 1.0 - df;
  }

#undef PAL_AB
#undef PAL_K
#undef PAL_CI
#undef PAL_CG

}; // namespace astroflow

#endif
