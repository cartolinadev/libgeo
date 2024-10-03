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

//#include <Eigen/Dense>


#include "utility/expect.hpp"
#include "boost/filesystem/path.hpp"
#include "jsoncpp/as.hpp"

namespace geo {

namespace landcover {

namespace ut = utility;
//namespace fs = boost::filesystem;

typedef ublas::matrix<float,ublas::row_major,
                      ublas::bounded_array<float, 4> > Matrix2;
typedef ublas::matrix<float,ublas::row_major,
                      ublas::bounded_array<float, 9> > Matrix3;


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
        //Eigen::Matrix<float,2,2> limits;

        Matrix2 limits2;

        // the actual function, each column has coefficients for one output
        // variable. Order RGB
        //Eigen::Matrix<float,3,3> model;

        Matrix3 model2;


        cv::Vec3b apply(const short elevation, const int precipitation) const {

/*            Eigen::Matrix<float,1,3>
                input(1,
                    math::clamp((float) elevation, limits(0,0), limits(0,1)),
                    math::clamp((float) precipitation, limits(1,0), limits(1,1))
                ), output;

            output = input * model; */

            float elev_ = math::clamp(
                (float) elevation, limits2(0,0), limits2(0,1));
            float precip_ = math::clamp(
                (float) precipitation, limits2(1,0), limits2(1,1));

            std::array<float,3> output {
                model2(0,0) + elev_ * model2(1,0) + precip_ * model2(2,0),
                model2(0,1) + elev_ * model2(1,1) + precip_ * model2(2,1),
                model2(0,2) + elev_ * model2(1,2) + precip_ * model2(2,2)
            };

            return cv::Vec3b(
                    // we clamp to 1, to reserve 0 for nodata if needed
                    math::clamp(std::round(output[2]), 1.f, 255.f),
                    math::clamp(std::round(output[1]), 1.f, 255.f),
                    math::clamp(std::round(output[0]), 1.f, 255.f));
        }

        Function()
            : limits2(2,2), model2(3,3) {};

    };

    // bivariate functions by landcover class
    std::unordered_map<uchar, Function> functions;
};


BivariateLandcover::BivariateLandcover()
    : detail(std::make_unique<Detail>()) {
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
    ut::expect(precipitation.type() == CV_32SC1, "Invalid precipitation data type.");

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
        const int *precipitationPtr = precipitation.ptr<int>(i);
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
                //class_.limits(i,j) = std::stof(token);
                class_.limits2(i,j) = std::stof(token);
            }

        for (int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++) {
                if (! std::getline(ss, token, ',')) {
                    throw std::runtime_error("Parsing error: model");
                }
                //class_.model(i,j) = std::stof(token);
                class_.model2(i,j) = std::stof(token);
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

// class ClassDefinition

Classes fromJson(const Json::Value &object) {

    Classes retval;

    for (const auto& jclass: object) {

        ClassDefinition class_;

        // compulsory elements
        get(class_.id, jclass, "id");
        get(class_.name, jclass, "name");

        // optional elements
        getOpt(class_.zIndex, jclass, "z-index");
        getOpt(class_.expectedLuma, jclass, "expected-luma");
        getOpt(class_.specularReflectivity, jclass, "specular-reflectivity");
        getOpt(class_.shininess, jclass, "shininess");
        getOpt(class_.isFlat, jclass, "isFlat");

        // add class
        retval[class_.id] = class_;
    }

    return retval;
}

imgproc::RasterMask flatMask(const cv::Mat& landcover,
                            const Classes& classdef) {

    imgproc::RasterMask mask(landcover.cols, landcover.rows,
                             imgproc::RasterMask::EMPTY);

    // iterate through landcover, checking for pixels with isFlat classes
    for (int i = 0; i < landcover.rows; i++) {

        auto row = landcover.ptr<uchar>(i);

        for (int j = 0; j < landcover.cols; j++) {

            auto it = classdef.find(*row);
            if (it != classdef.end() && it->second.isFlat) mask.set(j, i);

            row++;
        }
    }

    // done
    return mask;
}

imgproc::RasterMask inversionMask(const cv::Mat &landcover,
                                 const Classes& classdef) {

    /** the 3x3 moving window */
    class Window {

    public:
        // initialize window centered at 1,1 //
        Window(const cv::Mat &mat, const Classes& classdef)
            : mat(mat), classdef(classdef) { row(1); }

        // move window to the beginning of the row
        void row(int i) {

            row0 = mat.ptr<uchar>(i - 1);
            row1 = mat.ptr<uchar>(i);
            row2 = mat.ptr<uchar>(i + 1);
        }

        // move window 1px to the right
        void incx() {
            row0++; row1++; row2++;
        }

        void updateCell() {

            auto& cct(currentCellType);

            cct = CellType::neutral;

            std::set<uchar> wclasses;

            for (int i = 1; i <= 9; i++) wclasses.insert(v(i));

            if (wclasses.size() != 2) return;

            auto a = classdef.at(*wclasses.begin()),
                 b = classdef.at(*wclasses.rbegin());

            if ((a.zIndex - b.zIndex) * (a.expectedLuma - b.expectedLuma) > 0)
                cct = CellType::regular;
            else
                cct = CellType::inverted;

        }

        bool regular() const {
            return (currentCellType == CellType::regular);
        }

        bool inverted() const {
            return (currentCellType == CellType::inverted);
        }

    private:

        enum class CellType { neutral, regular, inverted };

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


        CellType currentCellType;
        const cv::Mat &mat;
        const uchar *row0, *row1, *row2;
        const Classes& classdef;
    };


    // initialize, sanity
    auto& lc(landcover);

    ut::expect(landcover.type() == CV_8UC1);
    ut::expect(lc.cols >= 3 && lc.rows >= 3);

    Window window(lc, classdef);

    // find regular and inverted 2-class boundaries
    auto regularMap = cv::Mat(lc.rows, lc.cols, CV_8U, cv::Scalar(255));
    auto invertedMap = cv::Mat(lc.rows, lc.cols, CV_8U, cv::Scalar(255));

    for (int i = 1; i < lc.rows - 1; i++) {

        window.row(i);

        for (int j = 1; j < lc.cols - 1; j++) {

            window.updateCell();
            if (window.regular()) regularMap.at<uchar>(i, j) = 0;
            if (window.inverted()) invertedMap.at<uchar>(i, j) = 0;

            // next col
            window.incx();
        }
    }

    //cv::imwrite("regular.png", regularMap);
    //cv::imwrite("inverted.png", invertedMap);

    // distance transform
    cv::Mat distanceR, distanceI;

    cv::distanceTransform(regularMap, distanceR, cv::DIST_L1, 3, CV_8U);
    cv::distanceTransform(invertedMap, distanceI, cv::DIST_L1, 3, CV_8U);

    cv::Mat im(landcover.rows, landcover.cols, CV_8UC1, cv::Scalar(0));

    /*for (int i = 0; i < im.rows; i++)
        for (int j = 0; j < im.cols; j++)
            im.at<uchar>(i,j)
                = distanceI.at<uchar>(i,j) < distanceR.at<uchar>(i,j) ? 0xff: 0;

    cv::imwrite("mask.png", im); */

    // the mask
    imgproc::RasterMask mask(landcover.cols, landcover.rows,
                imgproc::RasterMask::EMPTY);

    for (int i = 0; i < im.rows; i++)
        for (int j = 0; j < im.cols; j++) {

            if (distanceI.at<uchar>(i,j) < distanceR.at<uchar>(i,j))
                mask.set(j,i);
        }

    return mask;
}

} // namespace landcover

} // namespace geo
