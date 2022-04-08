#include "check_flip.hh"
#include <igl/predicates/predicates.h>
#include <igl/doublearea.h>

template <typename Scalar>
int check_flip(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &uv, const Eigen::MatrixXi &Fn, bool print_flip)
{
  using mpfr::mpreal;
  int fl = 0;
  mpfr::mpreal::set_default_prec(4*mpfr::mpreal::get_default_prec());
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> A;
  igl::doublearea(uv, Fn, A);
  for (int i = 0; i < Fn.rows(); i++)
  {
    Eigen::Matrix<Scalar, 1, 2> a(uv(Fn(i, 0), 0), uv(Fn(i, 0), 1));
    Eigen::Matrix<Scalar, 1, 2> b(uv(Fn(i, 1), 0), uv(Fn(i, 1), 1));
    Eigen::Matrix<Scalar, 1, 2> c(uv(Fn(i, 2), 0), uv(Fn(i, 2), 1));
    if (std::is_same<Scalar, mpreal>::value)
    {
      for(int k = 0; k < 2; k++){
        mpfr::mpreal _a = a(k); _a.set_prec(mpfr::mpreal::get_default_prec());
        mpfr::mpreal _b = b(k); _b.set_prec(mpfr::mpreal::get_default_prec());
        mpfr::mpreal _c = c(k); _c.set_prec(mpfr::mpreal::get_default_prec());
        a(k) = Scalar(_a); b(k) = Scalar(_b); c(k) = Scalar(_c);
      }
      Scalar signed_area = a(0) * b(1) - b(0) * a(1) +
                           b(0) * c(1) - c(0) * b(1) +
                           c(0) * a(1) - a(0) * c(1);
      if (signed_area < 0.0)
      {
        fl++;
        if (print_flip)
        {
          std::cout << "flip at triangle " << i << ": " << Fn.row(i) << "," << A(i) << std::endl;
        }
      }
    }
    else
    {
      Eigen::Matrix<double, 1, 2> a_db(uv(Fn(i, 0), 0), uv(Fn(i, 0), 1));
      Eigen::Matrix<double, 1, 2> b_db(uv(Fn(i, 1), 0), uv(Fn(i, 1), 1));
      Eigen::Matrix<double, 1, 2> c_db(uv(Fn(i, 2), 0), uv(Fn(i, 2), 1));
      if (igl::predicates::orient2d(a_db, b_db, c_db) != igl::predicates::Orientation::POSITIVE)
      {
        fl++;
        if (print_flip)
        {
          std::cout << "flip at triangle " << i << ": " << Fn.row(i) << "," << A(i) << std::endl;
        }
      }
    }
  }
  mpfr::mpreal::set_default_prec(mpfr::mpreal::get_default_prec()/4);
  return fl;
}

// explicit instantiation
template int check_flip<double>(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &, const Eigen::MatrixXi &, bool);
template int check_flip<mpfr::mpreal>(const Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> &, const Eigen::MatrixXi &, bool);
