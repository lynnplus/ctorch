// Copyright (c) 2023 Lynn <lynnplus90@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ctorch/ctorch.h>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iomanip>

using namespace std;

struct ObjectInfo {
    cv::Rect bbox_rect;
    float score;
    int class_id;
};

struct LetterboxInfo {
    int top_pad;
    int down_pad;
    int left_pad;
    int right_pad;
    float scale;
};

const int input_width = 640;
const int input_height = 640;
const float conf_threshold = 0.4;
const float iou_threshold = 0.5;
const int max_hw = 4096;

TorchDevice device = {TorchDeviceType_CPU};


void pre_process(const cv::Mat &in, cv::Mat &out, LetterboxInfo &letterboxInfo) {

    auto in_h = static_cast<float>(in.size().height);
    auto in_w = static_cast<float>(in.size().width);

    float scale = std::min(float(input_width) / in_w, float(input_height) / in_h);
    int new_unpad_w = int(std::round(in_w * scale));
    int new_unpad_h = int(std::round(in_h * scale));

    float dw = float(input_width - new_unpad_w) / 2.0f;
    float dh = float(input_height - new_unpad_h) / 2.0f;

    cv::resize(in, out, cv::Size(new_unpad_w, new_unpad_h));

    int top = int(std::round(dh - 0.1f));
    int down = int(std::round(dh + 0.1));
    int left = int(std::round(dw - 0.1));
    int right = int(std::round(dw + 0.1));

    letterboxInfo.left_pad = left;
    letterboxInfo.right_pad = right;
    letterboxInfo.top_pad = top;
    letterboxInfo.down_pad = down;
    letterboxInfo.scale = scale;

    cv::copyMakeBorder(out, out, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32FC3, 1.0 / 255.0);
}

//NMS
void non_maximum_suppression(const std::vector<ObjectInfo> &src, float confidence_threshold, float iou,
                             std::vector<int> &nms_indices) {
    std::vector<cv::Rect> offset_box_list;
    std::vector<float> score_list;

    for (const auto &item: src) {
        int offset = item.class_id * max_hw;

        cv::Point pt(offset, offset);
        cv::Rect temp(item.bbox_rect + pt);

        offset_box_list.emplace_back(temp);
        score_list.emplace_back(item.score);
    }

    cv::dnn::NMSBoxes(offset_box_list, score_list, confidence_threshold, iou, nms_indices);
}


void restore_bounding_box_size(const cv::Rect &src, const LetterboxInfo &letterboxInfo, const cv::Size &inputShape,
                               cv::Rect &dst) {
    auto clip = [](int n, int lower, int upper) {
        return std::max(lower, std::min(n, upper));
    };

    int x1 = (int) std::round(float(src.tl().x - letterboxInfo.left_pad) / letterboxInfo.scale);  // x padding
    int y1 = (int) std::round(float(src.tl().y - letterboxInfo.top_pad) / letterboxInfo.scale);  // y padding
    int x2 = (int) std::round(float(src.br().x - letterboxInfo.right_pad) / letterboxInfo.scale);  // x padding
    int y2 = (int) std::round(float(src.br().y - letterboxInfo.down_pad) / letterboxInfo.scale);  // y padding

    x1 = clip(x1, 0, inputShape.width);
    y1 = clip(y1, 0, inputShape.height);
    x2 = clip(x2, 0, inputShape.width);
    y2 = clip(y2, 0, inputShape.height);
    dst = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}


int detect(TorchModule module, const cv::Mat &input, float confidence_threshold, float iou,
           std::vector<ObjectInfo> &outputs) {

    cv::Mat img;
    LetterboxInfo letterboxInfo{};
    pre_process(input, img, letterboxInfo);


    TorchBlob blob = {img.data, 1, img.channels(), img.size().height, img.size().width};
    auto value = torch_module_forward_by_blob(module, &blob, &device, false);

    TorchStatus status;
    TensorResultBox *boxes = nullptr;
    int len = (int) torch_tensor_parse_to_bbox(value, confidence_threshold, -1, &boxes, &status);
    if (status.code != 0) {
        cerr << "detect fail:" << status.msg << endl;
        torch_status_clear(&status);
        torch_tensor_delete(value);
        return -1;
    }
    if (len <= 0) {
        torch_tensor_delete(value);
        return 0;
    }

    std::vector<ObjectInfo> data;
    data.reserve(len);

    for (int i = 0; i < len; ++i) {
        auto item = boxes[i];

        int ox = static_cast<int>(std::round(item.centerX - (item.width / 2.0f)));
        int oy = static_cast<int>(std::round(item.centerY - (item.height / 2.0f)));
        int sw = static_cast<int>(std::round(item.width));
        int sh = static_cast<int>(std::round(item.height));

        ObjectInfo o = {
                cv::Rect(ox, oy, sw, sh),
                item.score,
                item.class_idx
        };
        data.emplace_back(o);
    }
    std::free(boxes);
    torch_tensor_delete(value);

    std::vector<int> nms_indices;
    non_maximum_suppression(data, confidence_threshold, iou, nms_indices);

    cv::Size inputShape = input.size();
    for (const auto &i: nms_indices) {
        ObjectInfo item = data[i];
        restore_bounding_box_size(item.bbox_rect, letterboxInfo, inputShape, item.bbox_rect);
        outputs.emplace_back(item);
    }
    return 0;
}


void draw_bbox_to_target(const std::vector<ObjectInfo> &outputs, const vector<string> &names, const cv::Mat &mat) {

    for (const auto &object_info: outputs) {
        // Draws object bounding box
        cv::rectangle(mat, object_info.bbox_rect, cv::Scalar(0, 0, 255), 1);

        // Class info text
        std::string class_name = "unknown";

        if (names.size() > object_info.class_id) {
            class_name = names[object_info.class_id];
        }

        std::stringstream class_info;
        class_info << class_name << " " << std::fixed << std::setprecision(2) << object_info.score;

        // Size of class info text
        auto font_face = cv::FONT_HERSHEY_SIMPLEX;
        float font_scale = 0.4;
        int thickness = 1;
        int baseline = 0;
        cv::Size class_info_size = cv::getTextSize(class_info.str(), font_face, font_scale, thickness, &baseline);

        // Draws rectangle of class info text
        int height_offset = 5;  // [px]
        cv::Point class_info_top_left = cv::Point(object_info.bbox_rect.tl().x,
                                                  object_info.bbox_rect.tl().y - class_info_size.height -
                                                  height_offset);
        cv::Point class_info_bottom_right = cv::Point(object_info.bbox_rect.tl().x + class_info_size.width,
                                                      object_info.bbox_rect.tl().y);


        cv::rectangle(mat, class_info_top_left, class_info_bottom_right, cv::Scalar(0, 0, 255), -1);

        // Draws class info text
        cv::Point class_info_text_position = cv::Point(object_info.bbox_rect.tl().x,
                                                       object_info.bbox_rect.tl().y - height_offset);
        cv::putText(mat, class_info.str(), class_info_text_position,
                    font_face, font_scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
    }
}

void load_class_name(const string &path, vector<string> &names) {
    std::ifstream class_name_ifs(path);
    if (class_name_ifs.is_open()) {
        std::string class_name;
        while (std::getline(class_name_ifs, class_name)) {
            names.emplace_back(class_name);
        }
        class_name_ifs.close();
    } else {
        cerr << "open class name file fail" << endl;
    }
}

int main() {

    const string input = "input.jpeg";
    const string modelPath = "yolov5s.torchscript";
    const string namesPath = "coco.names";
    const string windowName = "Result";

    cv::Mat inputMat = cv::imread(input);
    if (inputMat.empty()) {
        cerr << "input mat empty" << endl;
        return 1;
    }

    TorchStatus status;
    auto module = torch_module_load(modelPath.c_str(), &status);
    if (status.code != 0) {
        cerr << "load model fail:" << status.msg << endl;
        torch_status_clear(&status);
        return 1;
    }

    vector<string> names;
    vector<ObjectInfo> outputs;
    //Empty inferences to warm up
    try {
        cv::Mat tmp_image = cv::Mat::zeros(input_height, input_width, CV_32FC3);
        int res = detect(module, tmp_image, 1.0, 1.0, outputs);

        if (res != 0) {
            torch_module_delete(module);
            cerr << "warm up fail!" << endl;
            return 1;
        }

        outputs.clear();
        res = detect(module, inputMat, conf_threshold, iou_threshold, outputs);

        if (res != 0) {
            torch_module_delete(module);
            cerr << "detect fail!" << endl;
            return 1;
        }

        torch_module_delete(module);

        load_class_name(namesPath, names);


        draw_bbox_to_target(outputs, names, inputMat);
    } catch (std::exception &e) {
        cerr << "err:" << e.what() << endl;
        return 1;
    }

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, inputMat);
    cv::waitKey(0);
    return 0;
}