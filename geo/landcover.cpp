/**
 * Copyright (c) 2017 Melown Technologies SE
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
/*
 * @file landcover.cpp
 */


#include "landcover.hpp"

#include <unordered_map>

#include <Eigen/Dense>

#include "utility/expect.hpp"
#include "boost/filesystem/path.hpp"

namespace geo {

namespace ut = utility;
namespace fs = boost::filesystem;

typedef imgproc::quadtree::RasterMask RasterMask;

struct BivariateLandcover::Detail {

    struct Function {

        // class id
        uchar id;

        // class name
        std::string name;

        // each row contains min and max values for one variable,
        // defining the range where the model is defined.
        // input values are clamped to these limits when the model is applied.
        Eigen::Matrix<float,2,2> limits;

        // the actual function, each column has coefficients for one output
        // variable. Order RGB
        Eigen::Matrix<float,3,3> model;

        cv::Vec3b apply(const short elevation, const short precipitation) const {

            Eigen::Matrix<float,1,3>
                input(1,
                    math::clamp((float) elevation, limits(0,0), limits(0,1)),
                    math::clamp((float) precipitation, limits(1,0), limits(1,1))
                ), output;

            output = input * model;

            return cv::Vec3b(
                    // we clamp to 1, to reserve 0 for nodata if needed
                    math::clamp(std::round(output(2)), 1.f, 255.f),
                    math::clamp(std::round(output(1)), 1.f, 255.f),
                    math::clamp(std::round(output(0)), 1.f, 255.f));
        }

    };

    // bivariate functions by landcover class
    std::unordered_map<uchar, Function> functions;
};


BivariateLandcover::BivariateLandcover() {
    std::make_unique<Detail>();
}


BivariateLandcover::~BivariateLandcover() = default;

BivariateLandcover & BivariateLandcover::operator = (
    BivariateLandcover && ) noexcept = default;

BivariateLandcover::BivariateLandcover(BivariateLandcover &&) noexcept = default;


cv::Mat BivariateLandcover::apply(
    const cv::Mat & landcover, const cv::Mat & elevation,
    const cv::Mat & precipitation, Mask& mask) {

    // sanity
    ut::expect(landcover.type() == CV_8UC1, "Invalid landcover data type.");
    ut::expect(elevation.type() == CV_16SC1, "Invalid elevation data type.");
    ut::expect(precipitation.type() == CV_16SC1, "Invalid precipitation data type.");

    ut::expect(landcover.size() == elevation.size()
        && landcover.size() == precipitation.size(), "invalid data size");

    ut::expect(mask.size().width == landcover.cols
        && mask.size().height == landcover.rows, "Invalid mask size.");

    // init
    cv::Mat retval(landcover.size(), CV_8UC3, cv::Scalar(0,0,0));

    // process
    for (int i = 0; i < retval.rows; i++) {

        const uchar *landcoverPtr = landcover.ptr<uchar>(i);
        const short *elevationPtr = elevation.ptr<short>(i);
        const short *precipitationPtr = precipitation.ptr<short>(i);
        cv::Vec3b *retvalPtr = retval.ptr<cv::Vec3b>(i);

        for (int j = 0; j < retval.cols; j++, landcoverPtr++, elevationPtr++,
            precipitationPtr++, retvalPtr++) {

            if (!mask.get(j,i)) continue;

            auto it = detail->functions.find(*landcoverPtr);

            if ( it == detail->functions.end()) {
                mask.set(j, i, false); continue;
            }

            *retvalPtr = it->second.apply(*elevationPtr, *precipitationPtr);
        }
    }

    // return
    return retval;
}


BivariateLandcover BivariateLandcover::loadCsv(
    const boost::filesystem::path & path) {

    BivariateLandcover result;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    std::string line;
    while (std::getline(file, line)) {

        std::stringstream ss(line);
        std::string token;

        Detail::Function class_;

        std::getline(ss, token, ',');
        class_.id = std::stoi(token);

        std::getline(ss, token, ',');
        class_.name = token;

        for (int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++) {
                if (! std::getline(ss, token, ',')) {
                    throw std::runtime_error("Parsing error: limits");
                }
                class_.limits(i,j) = std::stof(token);
            }

        for (int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++) {
                if (! std::getline(ss, token, ',')) {
                    throw std::runtime_error("Parsing error: model");
                }
                class_.model(i,j) = std::stof(token);
            }

        if (std::getline(ss, token, ',')) {
            throw std::runtime_error("Excessive data on line");
        }

        result.detail->functions[class_.id] = class_;

    }

    file.close();

    // doone
    return result;
}



} // namespace geo
