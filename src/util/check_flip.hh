#ifndef CHECK_FLIP_H
#define CHECK_FLIP_H

#include <iostream>
#include <vector>
#include <unsupported/Eigen/MPRealSupport>

template <typename Scalar>
int check_flip(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &uv, const Eigen::MatrixXi &Fn, bool print_flip = false);

#endif