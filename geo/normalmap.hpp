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

#ifndef geo_normalmaps_hpp_included
#define geo_normalmaps_hpp_included

#include <math/math_all.hpp>
#include <utility/expect.hpp>
#include <geo/csconvertor.hpp>

#include <opencv2/opencv.hpp>

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
 * @brief create a normal map from a dem with given resolution
 * @param dem input dem, expected to be a single channel 64-bit float matrix
 * @param pixelSize size of pixel in input DEM
 * @param viewspaceRF if true, output normals will be in view space (x axis
 *  upward, z axis towards the viewer). If false, image space (y axis downward,
 *  z axis away from the viewer) is used.
 * @return the resultant three channel normal map, which is two pixels shorter
 *  from the input on each side: the normal map is computed only for internal
 *  pixels. The normals are normalized with their x, y and z values stored
 *  in channels 0, 1 and 2 respectively. The resultant matrix type is CV_32FC3.
 */
cv::Mat demNormals(
    const cv::Mat& dem, const math::Size2f& pixelSize,
    const Algorithm& algorithm = Algorithm::zevenbergenThorne,
    const bool viewspaceRf = true);

/**
 * @brief convert a normal map to a different spatial reference.
 * @details For a normal map genereated via demToNormalMap, this function
 * converts normals from one (typically projected, taken from the original DEM)
 * to another (typically physical) spatial reference system. Conversion is
 * performed in place.
 *
 * @param normalMap the normal map for in-place conversion
 * @param extents extents in source SRS
 * @param convertor the convertor
 * @param linearOptimization if true, the supplied convertor is used only for
 *  corners of the matrix, for the rest of the pixels a linear conversion is
 *  performed using bilinear interpolation. This makes sense unless the normal
 *  map spans more than haf a hemisphere.
 */

void convertNormals(cv::Mat &normalMap, const math::Extents2& extents,
    const CsConvertor& convertor, bool linearOptimization = true);

/**
 * Export a normal map to an 8-bit unsigned char RGB matrix. The components
 * of the normal are linearly scaled from the [-1.0, 1.0] range to [0.0-255.0].
 *
 * Note the BGR channel (OpenCV default).
 */
cv::Mat exportToBGR(const cv::Mat &normalMap);


/**
 * @brief Obtain a perefectly flat normal map. Used mainly for diagnostics.
 */

cv::Mat flatSurfaceNormals(const math::Size2& size,
                           const int depthType = CV_32F);


} // namespace normalmap

} // namespace geo


#endif // geo_normalmaps_hpp_included
