// Optional OpenCV backend for the improved CV detector (`--features opencv`).
// Does the per-pixel stages (blur, Canny, dilation, connected components) with
// OpenCV's SIMD/multithreaded code; the Rust side keeps the size filters and
// nesting rules, so targets match the pure-Rust path.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdint>

extern "C" int stochos_cv_components(
    const uint8_t *gray, int w, int h, double low, double high, int kw, int kh,
    int32_t *out, int max_out
) {
    try {
        cv::Mat src(h, w, CV_8UC1, const_cast<uint8_t *>(gray));
        cv::Mat blurred, edges, joined, labels, stats, centroids;
        // Same blur and L2 gradient magnitude as imageproc's Canny.
        cv::GaussianBlur(src, blurred, cv::Size(7, 7), 1.4, 1.4, cv::BORDER_REPLICATE);
        cv::Canny(blurred, edges, low, high, 3, true);
        cv::dilate(edges, joined, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kw, kh)));
        int n = cv::connectedComponentsWithStats(joined, labels, stats, centroids, 8, CV_32S);
        int count = 0;
        for (int i = 1; i < n && count < max_out; i++, count++) {  // label 0 is background
            out[count * 5 + 0] = stats.at<int>(i, cv::CC_STAT_LEFT);
            out[count * 5 + 1] = stats.at<int>(i, cv::CC_STAT_TOP);
            out[count * 5 + 2] = stats.at<int>(i, cv::CC_STAT_WIDTH);
            out[count * 5 + 3] = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            out[count * 5 + 4] = stats.at<int>(i, cv::CC_STAT_AREA);
        }
        return count;
    } catch (...) {
        return -1;
    }
}
