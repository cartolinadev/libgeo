/**
 * Copyright (c) 2024 Melown Technologies SE
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

/**
  * @file normalmap.hpp
  * @author Ondrej Prochazka <ondrej.prochazka@montevallo.cz>
  *
  * Normal-map and bump-map generation from DEMs and grayscale images.
 **/


#ifndef geo_normalmaps_hpp_included
#define geo_normalmaps_hpp_included

#include "srsdef.hpp"
#include <math/math_all.hpp>
#include <utility/expect.hpp>
#include <geo/csconvertor.hpp>
#include "geo/coordinates.hpp"
#include <utility/expect.hpp>
#include <imgproc/rastermask.hpp>

#include <opencv2/opencv.hpp>

#include "utility/expect.hpp"

namespace geo {

namespace normalmap {

/**
  * @brief the hillshade algorithm
  *
  * See https://observablehq.com/@sahilchinoy/a-faster-hillshader
  * or GDAL DEMProcessing source for explanation on Zevenbergen-Thorne
  * and Horn.
  *
  * 8 point regression minimizes mean square error of a plane passing
  * through the neighbouring points. The mean is weighted by the inverse
  * distance of the point from the center point.
  */
enum class Algorithm {
    zevenbergenThorne, horn, regression
};

/**
 * Parameters for demNormals.
 * @param algorithm the hillshade (normal derivation) algorithm
 * @param viewspaceRF if true, output normals will be in view space (x axis
 *  upward, z axis towards the viewer). If false, image space (y axis downward,
 *  z axis away from the viewer) is used.
 * @param invertRelief states if the relief should be inverted, for image
 *  sources this means that darker tones are taken to be front (default for
 *  bump maps)
 */

struct Parameters {

    Algorithm algorithm = Algorithm::zevenbergenThorne;
    bool viewspaceRf = true;
    bool invertRelief = false;
    float zFactor = 1;
};


/**
 * Create a normal map from a DEM or graycale image with given resolution.
 *
 * @param dem input dem, of the same underlying value type as the first template
 *  argument
 * @param pixelSize size of pixel in input DEM
 * @param params see above
 * @param flatMask pixels matching this mask always yield upwards normal
 * @param inversionMask pixels matching this mask are relief inverted, the
 *  inversion is applied cumulatively to optional image-wid inversion as
 *  specified by params.
 * @return the resultant three channel normal map, which is two pixels shorter
 *  from the input on each side: the normal map is computed only for internal
 *  pixels. The normals are normalized with their x, y and z values stored
 *  in channels 0, 1 and 2 respectively. The resultant matrix type is CV_32FC3.
 */

template <typename value_type,
          typename Mask1 = imgproc::quadtree::RasterMask,
          typename Mask2 = imgproc::quadtree::RasterMask>
cv::Mat demNormals(
    const cv::Mat& dem, const math::Size2f& pixelSize,
    const Parameters& params,
    const Mask1& flatMask,
    const Mask2& inversionMask) {

    namespace ut = utility;

    /** the 3x3 moving window */
    class Window {

    public:
        // initialize window centered at 1,1 //
        Window(const cv::Mat &mat, const math::Size2f& pixelSize,
            const float zFactor)
            : mat(mat), pixelSize(pixelSize), zFactor(zFactor) {

            row(1);
        }

        // move window to the beginning of the row
        void row(int i) {

            row0 = mat.ptr<value_type>(i - 1);
            row1 = mat.ptr<value_type>(i);
            row2 = mat.ptr<value_type>(i + 1);
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

            double a = zFactor * 0.5 / psx * (v(6) - v(4));
            double b = zFactor * 0.5 / psy * (v(8) - v(2));

            return normalize(math::Point3(-a, -b, -1));
        }

        math::Point3 horn() const {

            float psx(pixelSize.width), psy(pixelSize.height);

            double a = zFactor * 0.125 / psx
                * (v(3) + 2 * v(6) + v(9) - v(1) - 2 * v(4) - v(7));
            double b = zFactor * 0.125 / psy
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

            //LOGONCE(debug) << abd;
            return normalize(math::Point3(
                zFactor * -abd[0], zFactor * -abd[1], -1));
        }

    private:
        const cv::Mat &mat;
        const value_type *row0, *row1, *row2;
        math::Size2f pixelSize;
        float zFactor;
    };


    int width{dem.cols}, height{dem.rows};

    // expect a single gray channel
    if (std::is_same<value_type, float>::value)
        ut::expect(dem.type() == CV_32FC1);
    if (std::is_same<value_type, double>::value)
        ut::expect(dem.type() == CV_64FC1);
    if (std::is_same<value_type, uchar>::value)
        ut::expect(dem.type() == CV_8UC3);

    ut::expect(width >= 3 && height >= 3);

    // empty map
    cv::Mat ret = cv::Mat::zeros(height - 2, width - 2,  CV_32FC3);

    // transformer
    Window window(dem, pixelSize, params.zFactor);

    for (int j = 0; j < height - 2; j++) {

        // start row
        window.row(j + 1);

        for (int i = 0; i < width - 2; i++ ) {

            math::Point3 normal;

            if (flatMask.get(i + 1, j + 1)) {

                // flat area
                normal = {0, 0, -1};

            } else {

                // compute normal
                switch (params.algorithm) {

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
            }

            // optionally invert relief (flip dx and dy)
            if (params.invertRelief) {
                normal[0] = - normal[0]; normal[1] = - normal[1];
            }

            if (inversionMask.get(i + 1, j + 1)) {
                normal[0] = - normal[0]; normal[1] = - normal[1];
            }

            // the normal is in image space,
            // we flip y and z to transform to view space if requested
            if  (params.viewspaceRf) {
                normal[1] = - normal[1]; normal[2] = - normal[2];
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

/**
 * @brief Create a normal map from a DEM or grayscale image. This is a 
 *  simplified version of the above function, which does not take masks and 
 *  uses default parameters.
 *
 * @param dem input dem or grayscale image
 * @param pixelSize size of pixel in input DEM
 * @param params normal map generation parameters
 * @return the resultant three channel normal map
 */
template <typename value_type,
          typename Mask1 = imgproc::quadtree::RasterMask,
          typename Mask2 = imgproc::quadtree::RasterMask>
cv::Mat demNormals(
    const cv::Mat& dem, const math::Size2f& pixelSize,
    const Parameters& params = Parameters()) {

    return demNormals<value_type>(dem, pixelSize, params,
        Mask1(dem.cols, dem.rows, Mask1::EMPTY),
        Mask2(dem.cols, dem.rows, Mask1::EMPTY));
}

namespace detail {

struct NoOpExtraConvertor {

    math::Matrix3 operator()(const math::Point3&) const {
        return boost::numeric::ublas::identity_matrix<double>(3);
    }
};

template <typename T = NoOpExtraConvertor>
void convertNormalsLinear(cv::Mat &normalMap, const CsConvertor &conv,
    const CsConvertor &iconv, const math::Matrix4 &raster2geo,
    const T& extraConv = {}) {

     LOG(debug) << "Converting normals with linear optimization. Size: "
         << normalMap.rows << "x" << normalMap.cols;

    using namespace math;

    auto& nm(normalMap);

    // obtain first order linearizations in matrix corners
    // raster2geo is affine, so the product's homogeneous coordinate stays 1;
    // we drop it with subrange rather than euclidian (no division needed).
    auto phys00 = conv(
        Point3(subrange(prod(raster2geo, Point4{0., 0., 0., 1.}), 0, 3)));
    auto phys01 = conv(
        Point3(subrange(prod(raster2geo, Point4{nm.cols - 1., 0., 0., 1.})
                        , 0, 3)));
    auto phys10 = conv(
        Point3(subrange(prod(raster2geo, Point4{0., nm.rows - 1., 0., 1.})
                        , 0, 3)));
    auto phys11 = conv(
        Point3(subrange(prod(raster2geo
                             , Point4{nm.cols - 1., nm.rows - 1., 0., 1.})
                        , 0, 3)));

    auto m00 = Matrix3(subrange(trans(iconv.linearize(phys00)), 0, 3, 0, 3));
    auto m01 = Matrix3(subrange(trans(iconv.linearize(phys01)), 0, 3, 0, 3));
    auto m10 = Matrix3(subrange(trans(iconv.linearize(phys10)), 0, 3, 0, 3));
    auto m11 = Matrix3(subrange(trans(iconv.linearize(phys11)), 0, 3, 0, 3));

    m00 = prod(extraConv(phys00), m00);
    m01 = prod(extraConv(phys01), m01);
    m10 = prod(extraConv(phys10), m10);
    m11 = prod(extraConv(phys11), m11);

    // iterate through matrix cells
    for (int i = 0; i < nm.rows; i++) {

        // start and end of row trafo
        float posy = (float) i / (nm.rows - 1);

        Matrix3 mi0 = (1 - posy) * m00 + posy * m10;
        Matrix3 mi1 = (1 - posy) * m01 + posy * m11;

        //LOGONCE(debug) << "mi0: " << mi0;
        //LOGONCE(debug) << "mi1: " << mi1;

        for (int j = 0; j < nm.cols; j++) {

            // local trafo
            float posx = (float) j / (nm.cols - 1);

            Matrix3 m = (1 - posx) * mi0 + posx * mi1;

            //LOGONCE(debug) << "local matrix:" << m;

            // transform in place
            cv::Vec3f on = nm.at<cv::Vec3f>(i, j);

            Point3 oldNormal(on(0), on(1), on(2));
            //LOGONCE(debug) << "old normal: " << oldNormal;

            Point3 normal = math::normalize(prod(m, oldNormal));
            //LOGONCE(debug) << "normal: " << normal;

            nm.at<cv::Vec3f>(i, j) = cv::Vec3f(
                normal[0], normal[1], normal[2]);

        }
    }

    // done
}

} // namespace detail


/**
 * @brief Convert a normal map to a different spatial reference.
 *
 * For a normal map genereated via demToNormalMap, this function
 * converts normals from one (typically projected, taken from the original DEM)
 * to another (typically physical) spatial reference system. Conversion is
 * performed in place and the normal map is expected to be of type CV_32FC3.
 *
 * The template  parameter T allows to specify an extra linear convertor, applied 
 * to the normals after the coordinate transformation. The convertor is passed
 * the physical coordinates of the pixel as an argument and is epxcted to return
 * a math::Matrix3. 
 *
 * @param normalMap the normal map for in-place conversion
 * @param extents extents in source SRS
 * @param conv convertor from source to target SRS
 * @param linearOptimization if true, the supplied convertor is used only for
 *  corners of the matrix, for the rest of the pixels a linear conversion is
 *  performed using bilinear interpolation. This makes sense unless the normal
 *  map spans more than haf a hemisphere.
 * @param extraConv an extra convertor applied to the normals after the coordinate
 *  transformation. This can be used to apply a space-dependent linear transform
 *  to the normals, for example to transform them to a tangential plane defined 
 *  by the local surface normal.      
 */

template <typename T = detail::NoOpExtraConvertor>
void convertNormals(cv::Mat &normalMap, const math::Extents2& extents,
    const geo::CsConvertor& conv, const T& extraConv = {}, 
    bool linearOptimization = true) {

    namespace ut = utility;

    //LOG(debug) << (linearOptimization
    //    ? "Converting normals with linear optimization."
    //    : "Converting normals without optimization.");

    auto& nm(normalMap);

    // we work only 32bit floats
    ut::expect(nm.type() == CV_32FC3, "3-channel 32bit matrix expected");

    // init transformations
    math::Size2f pxSize(
        (extents.ur[0] - extents.ll[0]) / nm.cols,
        (extents.ur[1] - extents.ll[1]) / nm.rows
    );

    auto iconv = conv.inverse();
    math::Matrix4 raster2geo = geo::raster2geo(extents, pxSize);

    // optimized conversion
    if (linearOptimization) {
        detail::convertNormalsLinear<T>(nm, conv, iconv, raster2geo, extraConv);
        return;
    }

    // non optimized conversion    
    for (int i = 0; i < nm.rows; i++)
        for (int j = 0; j < nm.cols; j++) {

            // local linear transformation (affine: drop homogeneous coord)
            auto phys = conv(math::Point3(subrange(prod(raster2geo,
                math::Point4{(double) j, (double) i, 0., 1.}), 0, 3)));

            auto m = math::Matrix4(trans(iconv.linearize(phys)));

            LOGONCE(debug) << "m: " << m;

            // transform in place
            cv::Vec3f on = nm.at<cv::Vec3f>(i, j);

            math::Point3 oldNormal(on(0), on(1), on(2));

            math::Point3 normal
                = math::normalize(prod(subrange(m, 0, 3, 0, 3), oldNormal));

            math::Point3 normalws = prod(extraConv(phys), normal);
            //LOGONCE(debug) << "normalws: " << normalws;

            nm.at<cv::Vec3f>(i, j) = cv::Vec3f(
                normalws[0], normalws[1], normalws[2]);

    }

    // done
}


/**
 * Convert normal map vectors by an orthonormal basis transform.
 *
 * Conversion is performed in place and the normal map is expected to be of
 * type CV_32FC3. The supplied matrix is assumed to be orthonormal and is
 * interpreted as a basis-change/rotation transform for normals, not as a
 * generic linear transformation.
 *
 * @param normalMap the normal map for in-place conversion
 * @param orthonormalTransform orthonormal basis transform applied to every
 *  normal in the map
 */
void convertNormals(cv::Mat &normalMap
                    , const math::Matrix3 &orthonormalTransform);

/**
 * Encode a normal map using octahedron 2-channel encoding.
 *
 * The map is converted in place and the RG values are stored in first and
 * second channels, with third channel optionally nullified.
 *
 * normalMap is expected to be of type CV_32FC3, and the normals are expected
 * to be unit vectors.
 *
 * @param normalMap normal map, encoded in place
 */

void encodeOct(cv::Mat &normalMap);


/**
 * Export a normal map to an 8-bit unsigned char RGB matrix. The components
 * of the normal are linearly scaled from the [-1.0, 1.0] range to [0.0-255.0].
 *
 * Normal map is expected to be of type CV_32FC3.
 *
 * Note the BGR channel (OpenCV default).
 */
cv::Mat exportToBGR(const cv::Mat &normalMap);


/**
 * Obtain a perfectly flat normal map with (0,0,1) values.
 * Used mainly for diagnostics.
 */

cv::Mat flatSurfaceNormals(const math::Size2& size,
                           const int depthType = CV_32F);


} // namespace normalmap

} // namespace geo


#endif // geo_normalmaps_hpp_included
