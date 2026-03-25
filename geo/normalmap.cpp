/**
 * Copyright (c) 2024 Ondrej Prochazka
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * *  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "normalmap.hpp"


namespace ut = utility;

namespace geo { namespace normalmap {


void convertNormals(cv::Mat &normalMap
                    , const math::Matrix3 &orthonormalTransform)
{
    auto &nm(normalMap);

    ut::expect(nm.type() == CV_32FC3, "3-channel 32bit matrix expected");

    for (int i = 0; i < nm.rows; ++i) {
        for (int j = 0; j < nm.cols; ++j) {
            const auto on(nm.at<cv::Vec3f>(i, j));
            const math::Point3 oldNormal(on(0), on(1), on(2));
            const math::Point3 normal(prod(orthonormalTransform, oldNormal));

            nm.at<cv::Vec3f>(i, j)
                = cv::Vec3f(normal[0], normal[1], normal[2]);
        }
    }
}

void encodeOct(cv::Mat &normalMap) {

    auto& nm(normalMap);

    // we need the "sign not zero" which, unlike math::sgn, is 1 at 0
    auto sgn = [](float arg) {
        return arg >= 0.f ? 1 : -1;
    };

    constexpr float Eps = 1e-08;

    // we work only 32bit floats
    ut::expect(nm.type() == CV_32FC3, "3-channel 32bit matrix expected");

    for (int i = 0; i < nm.rows; i++)
        for (int j = 0; j < nm.cols; j++) {

            auto& normal = nm.at<cv::Vec3f>(i, j);

            // octahedron encoding
            // https://brashandplucky.com/2022/07/07/octahedron-unitvector-encoding.html

            auto l1n = cv::norm(normal, cv::NORM_L1);

            if (! std::isfinite(l1n) || l1n < Eps) {
                LOG(warn1) << "Invalid (non-unit) vector.";
                nm.at<cv::Vec3f>(i,j) = {0.f,0.f,0.f};
                continue;
            }

            normal /= l1n;

            if (normal[2] < 0.0f) {


                auto tmp = (1.0f - std::fabs(normal[1])) * sgn(normal[0]);
                normal[1] = (1.0f - std::fabs(normal[0])) * sgn(normal[1]);
                normal[0] = tmp;
            }

            normal[2] = 0.0f;
    }
}

cv::Mat exportToBGR(const cv::Mat &normalMap) {

    auto& nm(normalMap);

    // we work only 32bit floats
    ut::expect(nm.type() == CV_32FC3, "3-channel 32bit matrix expected");

    // initialize
    cv::Mat ret = cv::Mat(nm.rows, nm.cols, CV_8UC3);

    // work the values
    for (int i = 0; i < nm.rows; i++)
        for (int j = 0; j < nm.cols; j++) {

            cv::Vec3f normal = nm.at<cv::Vec3f>(i, j);

            ret.at<cv::Vec3b>(i, j) =
                cv::Vec3b(
                    round(255 * 0.5 * (normal[2] + 1)),
                    round(255 * 0.5 * (normal[1] + 1)),
                    round(255 * 0.5 * (normal[0] + 1)));
        }

    // done
    return ret;
}

cv::Mat flatSurfaceNormals(const math::Size2& size,
                           const int depthType) {
    return cv::Mat(size.height, size.width,
                   CV_MAKETYPE(depthType, 3), cv::Scalar(0.0, 0.0, 1.0));
}


} // namespace normalmap

} // namespace geo
