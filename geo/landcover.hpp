/**
 * Copyright (c) 2023 Melown Technologies SE
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
  * @file landcover.hpp
  * @author Ondrej Prochazka <ondrej.prochazka@montevallo.cz>
  *
  * Subroutines for processing of bivariate landcover grids.
  *
  */

#ifndef geo_landcover_hpp_included
#define geo_landcover_hpp_included

#include <memory>

#include <boost/optional.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>

#include <imgproc/rastermask.hpp>

#include "geodataset.hpp"

namespace geo {

/**
  * Defines BGR color for each landcover class as a linear function of two
  * indepent variables.
  *
  * The purpose of this class is two aid in automatic coloring of bivariate
  * landcover grids. The method is inspired by "Natural-color Maps via Coloring
  * of Bivariate Grid Data" (Darbyshire & Jenny, 2017).
  *
  * The bivariate functions can be established using manual techniques or via a
  * training dataset. Within production environment (such as the tms/landcover
  * driver in vts-mapproxy, the precomputed bivariate function definitions are
  * loaded from a CSV file.
  */

class BivariateLCColors {

public:
    typedef imgproc::quadtree::RasterMask Mask;

    /**
     * @brief Perform automatic grid coloring.
     * @param landcover a landcover matrix, with values corresponding to landcover
     *      classes. Type needs to be CV8U_C1.
     * @param elevation coregistered elevation matrix (DEM), type CV16S_C1
     * @param precipitation coregistered precipitation matrix (DEM), type
     *      CV16S_C1
     * @param mask Validity mask. Pixels not set on input are not processsed.
     *      Mask is a non-const reference, as it is updated by the function
     *      whenever a landcover class for which a bivariate function
     *      is unknown is encountered, possibly indicating further invalid
     *      pixels on return.
     * @return colored BGR grid, type CV8U_C3.
     */
    cv::Mat apply(const cv::Mat & landcover,
                  const cv::Mat & elevation,
                  const cv::Mat & precipitation,
                  Mask& mask);

    /**
     * @brief create bivariate function definition from a CSV file.
     */
    static BivariateLCColors loadCsv(const boost::filesystem::path & path);


    /**
     * @brief save bivariate function definition to a CSV file.
     */
    void saveCsv(const boost::filesystem::path &path);

private:
    BivariateLCColors();

    //class Detail;

    //std::unique_ptr<Detail> detail;
};

} // namespace geo


#endif
