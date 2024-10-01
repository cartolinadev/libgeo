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

#include "geo/coordinates.hpp"
#include "utility/expect.hpp"

namespace ut = utility;

namespace geo { namespace normalmap {

namespace {

void convertNormalsLinear(cv::Mat &normalMap, const math::Extents2& extents,
    const CsConvertor& convertor) {

    auto& nm(normalMap);

    // we work only 32bit floats
    ut::expect(nm.type() == CV_32FC3, "3-channel 32bit matrix expected");

    // obtain first order linearizations in matrix corners
    math::Size2f pxSize(
        (extents.ur[0] - extents.ll[0]) / nm.cols,
        (extents.ur[1] - extents.ll[1]) / nm.rows
    );

    math::Matrix4 raster2geo = geo::raster2geo(extents, pxSize);

    //LOGONCE(debug) << "extents: " << extents;
    //LOGONCE(debug) << "raster2geo: " << raster2geo;

    auto m00 = convertor.linearize(subrange(
        prod(raster2geo, math::Point4{0, 0, 0, 1}), 0, 3));
    auto m01 = convertor.linearize(subrange(
        prod(raster2geo, math::Point4{nm.cols - 1.0, 0, 0, 1}), 0, 3));
    auto m10 = convertor.linearize(subrange(
        prod(raster2geo, math::Point4{0, nm.rows -1.0, 0, 1}), 0, 3));
    auto m11 = convertor.linearize(subrange(
        prod(raster2geo, math::Point4{nm.cols - 1., nm.rows - 1., 0, 1}), 0, 3));

    // iterate through matrix cells
    for (int i = 0; i < nm.rows; i++) {

        // start and end of row trafo
        float posy = (float) i / (nm.rows - 1);

        math::Matrix4 mi0 = (1 - posy) * m00 + posy * m10;
        math::Matrix4 mi1 = (1 - posy) * m01 + posy * m11;

        //LOGONCE(debug) << "mi0: " << mi0;
        //LOGONCE(debug) << "mi1: " << mi1;

        for (int j = 0; j < nm.cols; j++) {

            // local trafo
            float posx = (float) j / (nm.cols - 1);

            math::Matrix4 m = (1 - posx) * mi0 + posx * mi1;

            //LOGONCE(debug) << "local matrix:" << m;

            // transform in place
            cv::Vec3f on = nm.at<cv::Vec3f>(i, j);

            math::Point3 oldNormal(on(0), on(1), on(2));
            //LOGONCE(debug) << "old normal: " << oldNormal;

            math::Point3 normal
                = math::normalize(prod(subrange(m, 0, 3, 0, 3), oldNormal));
            //LOGONCE(debug) << "normal: " << normal;

            nm.at<cv::Vec3f>(i, j) = cv::Vec3f(
                normal[0], normal[1], normal[2]);

        }
    }

    // done
}

} // namespace

void convertNormals(cv::Mat &normalMap, const math::Extents2& extents,
    const CsConvertor& convertor, bool linearOptimization) {

    auto& nm(normalMap);

    if (linearOptimization) {
        convertNormalsLinear(nm, extents, convertor);
        return;
    }

    // non optimized conversion
    math::Size2f pxSize(
        (extents.ur[0] - extents.ll[0]) / nm.cols,
        (extents.ur[1] - extents.ll[1]) / nm.rows
    );

    math::Matrix4 raster2geo = geo::raster2geo(extents, pxSize);

    for (int i = 0; i < nm.rows; i++)
        for (int j = 0; j < nm.cols; j++) {

            // local linear transformation
            auto m = convertor.linearize(subrange(
                prod(raster2geo, math::Point4{
                    (double) j, (double) i, 0, 1}), 0, 3));

            // transform in place
            cv::Vec3f on = nm.at<cv::Vec3f>(i, j);

            math::Point3 oldNormal(on(0), on(1), on(2));
            //LOGONCE(debug) << "old normal: " << oldNormal;

            math::Point3 normal
                = math::normalize(prod(subrange(m, 0, 3, 0, 3), oldNormal));
            //LOGONCE(debug) << "normal: " << normal;

            nm.at<cv::Vec3f>(i, j) = cv::Vec3f(
                normal[0], normal[1], normal[2]);

    }

    // done
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
