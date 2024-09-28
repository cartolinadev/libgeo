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

cv::Mat demNormals(
    const cv::Mat& dem, const math::Size2f& pixelSize,
    const Algorithm& algorithm,
    const bool viewspaceRf) {

    /** the 3x3 moving window */
    class Window {

    public:
        // initialize window centered at 1,1 //
        Window(const cv::Mat &mat, const math::Size2f& pixelSize)
            : mat(mat), pixelSize(pixelSize) {

            row(1);
        }

        // move window to the beginning of the row
        void row(int i) {

            row0 = mat.ptr<double>(i - 1);
            row1 = mat.ptr<double>(i);
            row2 = mat.ptr<double>(i + 1);
        }

        // move window 1px to the right
        void incx() {

            row0++; row1++; row2++;
        }

        /**
         * @param index takes values 1-9, the window organized as
         *       x
         *   ----------
         *   |Z1 Z2 Z3|
         * y |Z4 Z5 Z6|
         *   |Z7 Z8 Z9|
         *   ----------
         */
        double v(uint index) const {

            switch(index) {

                case 1: return row0[0];
                case 2: return row0[1];
                case 3: return row0[2];
                case 4: return row1[0];
                case 5: return row1[1];
                case 6: return row1[2];
                case 7: return row2[0];
                case 8: return row2[1];
                case 9: return row2[2];
                default: return 0; // never reached
            }
        }

        /* these functions work in image coords (z points downwards).
         * se https://observablehq.com/@sahilchinoy/a-faster-hillshader
         * for formulas on ZevenbergenThorne and Horn.
         */

        math::Point3 zevenbergenThorne() const {

            float psx(pixelSize.width), psy(pixelSize.height);

            double a = 0.5 / psx * (v(6) - v(4));
            double b = 0.5 / psy * (v(8) - v(2));

            //LOGONCE(debug) << a << " " << b;

            return normalize(math::Point3(-a, -b, -1));
        }

        math::Point3 horn() const {

            float psx(pixelSize.width), psy(pixelSize.height);

            double a =  0.125 / psx
                * (v(3) + 2 * v(6) + v(9) - v(1) - 2 * v(4) - v(7));
            double b = 0.125 / psy
                * (v(7) + 2 * v(8) + v(9) - v(1) - 2 * v(2) - v(3));

            return normalize(math::Point3(-a, -b, -1));
        }

        math::Point3 regression() const {

            ublas::matrix<float> X(9,3);
            ublas::vector<float> Z(9);

            // feature matrix
            float psx(pixelSize.width), psy(pixelSize.height);

            ublas::row(X, 0) = math::Point3(-psx, -psy, 1) * 0.25;
            ublas::row(X, 1) = math::Point3( 0,   -psy, 1) * 0.50;
            ublas::row(X, 2) = math::Point3( psx, -psy, 1) * 0.25;
            ublas::row(X, 3) = math::Point3(-psx,  0,   1) * 0.50;
            ublas::row(X, 4) = math::Point3( 0,    0,   1) * 1.00;
            ublas::row(X, 5) = math::Point3( psx,  0,   1) * 0.50;
            ublas::row(X, 6) = math::Point3(-psx,  psy, 1) * 0.25;
            ublas::row(X, 7) = math::Point3( 0,    psy, 1) * 0.50;
            ublas::row(X, 8) = math::Point3( psx,  psy, 1) * 0.25;

            // target vector
            Z(0) = v(1) * 0.25;
            Z(1) = v(2) * 0.50;
            Z(2) = v(3) * 0.25;
            Z(3) = v(4) * 0.50;
            Z(4) = v(5) * 1.00;
            Z(5) = v(6) * 0.50;
            Z(6) = v(7) * 0.25;
            Z(7) = v(8) * 0.50;
            Z(8) = v(9) * 0.25;

            // obtain result
            ublas::matrix<float> xtx = ublas::prod(ublas::trans(X), X);

            math::Point3f abd = ublas::prod(ublas::prod(
                math::matrixInvert(xtx), ublas::trans(X)), Z);

            LOGONCE(debug) << abd;

            return normalize(math::Point3(-abd[0], -abd[1], -1));
        }

    private:
        const cv::Mat &mat;
        const double *row0, *row1, *row2;
        math::Size2f pixelSize;
    };


    int width{dem.cols}, height{dem.rows};

    // expect a single gray channel, sanity check, load data
    ut::expect(dem.type() == CV_64FC1);
    ut::expect(width >= 3 && height >= 3);

    // empty map
    cv::Mat ret = cv::Mat::zeros(height - 2, width - 2,  CV_32FC3);

    // transformer
    Window window(dem, pixelSize);


    for (int j = 1; j < height - 2; j++) {

        // start row
        window.row(j);

        for (int i = 1; i < width - 2; i++ ) {

            math::Point3 normal;

            switch (algorithm) {

                case Algorithm::zevenbergenThorne:
                    normal = window.zevenbergenThorne();
                    break;

                case Algorithm::horn:
                    normal = window.horn();
                    break;

                case Algorithm::regression:
                    normal = window.regression();
                    break;
            }

            // the normal is in image space,
            // we flip y and z to transform to view space if requested
            if  (viewspaceRf) {
                normal[1] = - normal[1];
                normal[2] = - normal[2];
            }

            ret.at<cv::Vec3f>(j,i) =
                cv::Vec3f(normal[0], normal[1], normal[2]);

            // next col
            window.incx();
        }

        // next row
    }

    // all done
    return ret;
}


void convertNormals(cv::Mat &normalMap, const math::Extents2& extents,
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

        auto mi0 = (1 - posy) * m00 + posy * m10;
        auto mi1 = (1 - posy) * m01 + posy * m11;

        for (int j = 0; j< nm.cols;j++) {

            // local trafo
            float posx = (float) j / (nm.cols - 1);

            auto m = (1 - posx) * mi0 + posx * mi1;

            // transform in place
            cv::Vec3f on = nm.at<cv::Vec3f>(i, j);
            math::Point3 oldNormal(on(0), on(1), on(2));

            math::Point3 normal
                = math::normalize(prod(subrange(m, 0, 3, 0, 3), oldNormal));

            nm.at<cv::Vec3f>(i, j) = cv::Vec3f(
                normal[0], normal[1], normal[2]);

        }
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
    return cv::Mat(size.height, size.height,
                   CV_MAKETYPE(depthType, 3), cv::Scalar(0.0, 0.0, 1.0));
}


} // namespace normalmap

} // namespace geo
