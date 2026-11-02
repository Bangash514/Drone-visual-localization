// Date: 2026-11-02
// Change aim: C++ port of the offline ORB map index and fast drone query
// File: cpp/localize.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

const int TILE = 1024;
const int MARGIN = 128;
const int MIN_INLIERS = 8;
const double DEFAULT_SCALES[] = {0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0};

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

double round1(double v) { return std::round(v * 10.0) / 10.0; }
double round2(double v) { return std::round(v * 100.0) / 100.0; }

cv::Ptr<cv::ORB> makeOrb(int nfeatures) {
    return cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
}

struct Location {
    bool valid = false;
    int inliers = 0;
    bool convex = false;
    bool success = false;
    cv::Point2f center;
    std::vector<cv::Point2f> quad;
};

Location estimateLocation(const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1,
                          const cv::Size& droneSize, double thresh) {
    Location loc;
    if (p0.size() < 4) return loc;
    std::vector<uchar> mask;
    cv::Mat H = cv::findHomography(p0, p1, cv::RANSAC, thresh, mask);
    if (H.empty()) return loc;
    float w = static_cast<float>(droneSize.width), h = static_cast<float>(droneSize.height);
    std::vector<cv::Point2f> corners = {{0, 0}, {w, 0}, {w, h}, {0, h}};
    std::vector<cv::Point2f> c = {{w / 2.0f, h / 2.0f}};
    cv::perspectiveTransform(corners, loc.quad, H);
    std::vector<cv::Point2f> cc;
    cv::perspectiveTransform(c, cc, H);
    loc.center = cc[0];
    loc.inliers = 0;
    for (uchar m : mask) loc.inliers += m ? 1 : 0;
    loc.convex = cv::isContourConvex(loc.quad);
    loc.success = loc.convex && loc.inliers >= MIN_INLIERS;
    loc.valid = true;
    return loc;
}

struct IndexData {
    std::vector<cv::Point2f> xy;
    cv::Mat des;
    int width = 0, height = 0;
    double buildMs = -1.0;
};

void detectTiled(const cv::Mat& gray, int perTile, IndexData& out) {
    cv::Ptr<cv::ORB> orb = makeOrb(perTile);
    std::vector<cv::Mat> descs;
    int w = gray.cols, h = gray.rows;
    for (int y0 = 0; y0 < h; y0 += TILE) {
        for (int x0 = 0; x0 < w; x0 += TILE) {
            int x1 = std::min(x0 + TILE, w), y1 = std::min(y0 + TILE, h);
            int px0 = std::max(0, x0 - MARGIN), py0 = std::max(0, y0 - MARGIN);
            int px1 = std::min(w, x1 + MARGIN), py1 = std::min(h, y1 + MARGIN);
            cv::Mat tile = gray(cv::Rect(px0, py0, px1 - px0, py1 - py0)).clone();
            std::vector<cv::KeyPoint> kps;
            cv::Mat d;
            orb->detectAndCompute(tile, cv::noArray(), kps, d);
            if (d.empty() || kps.empty()) continue;
            cv::Mat kept;
            for (size_t i = 0; i < kps.size(); ++i) {
                float x = kps[i].pt.x + static_cast<float>(px0);
                float y = kps[i].pt.y + static_cast<float>(py0);
                if (x >= x0 && x < x1 && y >= y0 && y < y1) {
                    out.xy.emplace_back(x, y);
                    kept.push_back(d.row(static_cast<int>(i)));
                }
            }
            if (!kept.empty()) descs.push_back(kept);
        }
    }
    cv::vconcat(descs, out.des);
}

void writeIdx(const std::string& path, const IndexData& idx) {
    std::ofstream f(path, std::ios::binary);
    f.write("DVLIDX01", 8);
    int32_t hdr[3] = {idx.width, idx.height, static_cast<int32_t>(idx.xy.size())};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(&idx.buildMs), sizeof(double));
    f.write(reinterpret_cast<const char*>(idx.xy.data()),
            static_cast<std::streamsize>(idx.xy.size() * sizeof(cv::Point2f)));
    f.write(reinterpret_cast<const char*>(idx.des.data),
            static_cast<std::streamsize>(idx.xy.size() * 32));
}

bool readIdx(const std::string& path, IndexData& idx) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "DVLIDX01", 8) != 0) return false;
    int32_t hdr[3];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    f.read(reinterpret_cast<char*>(&idx.buildMs), sizeof(double));
    idx.width = hdr[0];
    idx.height = hdr[1];
    idx.xy.resize(static_cast<size_t>(hdr[2]));
    f.read(reinterpret_cast<char*>(idx.xy.data()),
           static_cast<std::streamsize>(idx.xy.size() * sizeof(cv::Point2f)));
    idx.des.create(hdr[2], 32, CV_8U);
    f.read(reinterpret_cast<char*>(idx.des.data), static_cast<std::streamsize>(hdr[2]) * 32);
    return static_cast<bool>(f);
}

struct QueryResult {
    Location loc;
    int droneKp = 0;
    int matches = 0;
    double detectMs = 0, matchMs = 0, homographyMs = 0;
    double scale = 1.0;
};

class MapIndex {
public:
    MapIndex(const IndexData& d, const std::string& kind) : data(d), kind_(kind) {
        Clock::time_point t = Clock::now();
        if (kind == "flann") {
            matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1),
                                                         cv::makePtr<cv::flann::SearchParams>(50));
        } else {
            matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING);
        }
        matcher->add(std::vector<cv::Mat>{data.des});
        matcher->train();
        loadMs = msSince(t);
    }

    QueryResult query(const cv::Mat& gray, int nfeatures, double ratio, double ransac) const {
        QueryResult r;
        Clock::time_point t = Clock::now();
        cv::Ptr<cv::ORB> orb = makeOrb(nfeatures);
        std::vector<cv::KeyPoint> kps;
        cv::Mat des;
        orb->detectAndCompute(gray, cv::noArray(), kps, des);
        r.detectMs = msSince(t);
        r.droneKp = static_cast<int>(kps.size());
        if (des.empty() || kps.size() < 4) return r;
        t = Clock::now();
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(des, knn, 2);
        std::vector<cv::Point2f> p0, p1;
        for (const auto& p : knn) {
            if (p.size() == 2 && p[0].distance < ratio * p[1].distance) {
                p0.push_back(kps[static_cast<size_t>(p[0].queryIdx)].pt);
                p1.push_back(data.xy[static_cast<size_t>(p[0].trainIdx)]);
            }
        }
        r.matches = static_cast<int>(p0.size());
        r.matchMs = msSince(t);
        t = Clock::now();
        if (p0.size() >= 4) r.loc = estimateLocation(p0, p1, gray.size(), ransac);
        r.homographyMs = msSince(t);
        return r;
    }

    const IndexData& data;
    double loadMs = 0;
    const std::string& kind() const { return kind_; }

private:
    std::string kind_;
    cv::Ptr<cv::DescriptorMatcher> matcher;
};

cv::Mat resizeScale(const cv::Mat& img, double scale) {
    if (std::fabs(scale - 1.0) < 1e-6) return img;
    cv::Mat out;
    cv::Size sz(std::max(1, static_cast<int>(std::lround(img.cols * scale))),
                std::max(1, static_cast<int>(std::lround(img.rows * scale))));
    cv::resize(img, out, sz, 0, 0, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::map<std::string, std::string> parseOpts(int argc, char** argv, int start) {
    std::map<std::string, std::string> o;
    for (int i = start; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) o[k] = argv[++i];
        else o[k] = "1";
    }
    return o;
}

std::string get(const std::map<std::string, std::string>& o, const std::string& k, const std::string& d) {
    auto it = o.find(k);
    return it == o.end() ? d : it->second;
}

int cmdBuildIndex(const std::map<std::string, std::string>& o) {
    std::string mapPath = get(o, "map", ""), out = get(o, "out", "index.idx");
    int perTile = std::stoi(get(o, "features-per-tile", "800"));
    Clock::time_point t0 = Clock::now();
    cv::Mat gray = cv::imread(mapPath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "cannot read map: " << mapPath << "\n";
        return 1;
    }
    double readMs = msSince(t0);
    IndexData idx;
    idx.width = gray.cols;
    idx.height = gray.rows;
    detectTiled(gray, perTile, idx);
    idx.buildMs = round1(msSince(t0));
    writeIdx(out, idx);
    std::printf("{\n  \"index_build_ms\": %.1f,\n  \"map_read_ms\": %.1f,\n  \"map_size\": [%d, %d],\n"
                "  \"n_keypoints\": %zu,\n  \"tile\": %d,\n  \"margin\": %d,\n  \"features_per_tile\": %d,\n"
                "  \"index_file\": \"%s\"\n}\n",
                idx.buildMs, round1(readMs), idx.width, idx.height, idx.xy.size(), TILE, MARGIN, perTile,
                out.c_str());
    return 0;
}

int cmdQuery(const std::map<std::string, std::string>& o) {
    Clock::time_point t = Clock::now();
    cv::Mat drone = cv::imread(get(o, "drone", ""), cv::IMREAD_GRAYSCALE);
    if (drone.empty()) {
        std::cerr << "cannot read drone image\n";
        return 1;
    }
    double loadDroneMs = msSince(t);
    IndexData data;
    if (!readIdx(get(o, "index", ""), data)) {
        std::cerr << "cannot read index (expected .idx written by this tool or localize.py --export-idx)\n";
        return 1;
    }
    std::string kind = get(o, "matcher", "bf");
    MapIndex index(data, kind);
    int nfeat = std::stoi(get(o, "features", "1500"));
    double ratio = std::stod(get(o, "ratio", "0.75"));
    double ransac = std::stod(get(o, "ransac-thresh", "5.0"));
    int repeat = std::max(1, std::stoi(get(o, "repeat", "1")));
    std::vector<double> scales = {1.0};
    if (o.count("drone-scale-rel")) scales = {std::stod(o.at("drone-scale-rel"))};
    else if (o.count("scale-search")) scales.assign(std::begin(DEFAULT_SCALES), std::end(DEFAULT_SCALES));

    std::vector<double> dets, mats, homs;
    QueryResult last;
    for (int rep = 0; rep < repeat; ++rep) {
        bool have = false;
        QueryResult best;
        double sumDet = 0, sumMat = 0, sumHom = 0;
        for (double sc : scales) {
            cv::Mat d = resizeScale(drone, sc);
            if (std::min(d.cols, d.rows) < 32) continue;
            QueryResult r = index.query(d, nfeat, ratio, ransac);
            r.scale = sc;
            sumDet += r.detectMs;
            sumMat += r.matchMs;
            sumHom += r.homographyMs;
            if (!have || r.loc.inliers > best.loc.inliers) {
                best = r;
                have = true;
            }
        }
        dets.push_back(sumDet);
        mats.push_back(sumMat);
        homs.push_back(sumHom);
        last = best;
    }
    double det = median(dets), mat = median(mats), hom = median(homs), total = det + mat + hom;
    std::ostringstream js;
    js.setf(std::ios::fixed);
    js.precision(2);
    js << "{\n  \"method\": \"orb-index\",\n  \"matcher\": \"" << index.kind() << "\",\n"
       << "  \"map_size\": [" << data.width << ", " << data.height << "],\n"
       << "  \"drone_size\": [" << drone.cols << ", " << drone.rows << "],\n"
       << "  \"index_keypoints\": " << data.xy.size() << ",\n"
       << "  \"drone_keypoints\": " << last.droneKp << ",\n"
       << "  \"drone_scale_applied\": " << last.scale << ",\n"
       << "  \"matches\": " << last.matches << ",\n"
       << "  \"inliers\": " << last.loc.inliers << ",\n"
       << "  \"quad_convex\": " << (last.loc.valid ? (last.loc.convex ? "true" : "false") : "null") << ",\n"
       << "  \"success\": " << (last.loc.success ? "true" : "false") << ",\n"
       << "  \"timing_ms\": {\n"
       << "    \"index_build_ms\": " << data.buildMs << ",\n"
       << "    \"index_load_ms\": " << round1(index.loadMs) << ",\n"
       << "    \"drone_load_ms\": " << round1(loadDroneMs) << ",\n"
       << "    \"query_detect_ms\": " << round1(det) << ",\n"
       << "    \"query_match_ms\": " << round1(mat) << ",\n"
       << "    \"homography_ms\": " << round2(hom) << ",\n"
       << "    \"total_query_ms\": " << round1(total) << "\n  },\n"
       << "  \"total_query_under_100ms\": " << (total < 100.0 ? "true" : "false");
    if (last.loc.valid) {
        js << ",\n  \"center_px_map\": [" << last.loc.center.x << ", " << last.loc.center.y << "],\n"
           << "  \"footprint_px_map\": [";
        for (size_t i = 0; i < last.loc.quad.size(); ++i)
            js << (i ? ", " : "") << "[" << last.loc.quad[i].x << ", " << last.loc.quad[i].y << "]";
        js << "]";
    }
    js << "\n}\n";
    std::cout << js.str();
    std::string outdir = get(o, "out", "");
    if (!outdir.empty()) {
        std::ofstream f(outdir + "/log_cpp.json");
        if (f) f << js.str();
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: localize build-index --map M --out F.idx [--features-per-tile N]\n"
                     "       localize query --index F.idx --drone D [--out DIR] [--matcher bf|flann]\n"
                     "                      [--features N] [--ratio R] [--ransac-thresh T]\n"
                     "                      [--drone-scale-rel S | --scale-search] [--repeat N]\n";
        return 1;
    }
    std::string cmd = argv[1];
    std::map<std::string, std::string> o = parseOpts(argc, argv, 2);
    if (cmd == "build-index") return cmdBuildIndex(o);
    if (cmd == "query") return cmdQuery(o);
    std::cerr << "unknown command: " << cmd << "\n";
    return 1;
}
