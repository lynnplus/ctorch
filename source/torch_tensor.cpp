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

#include "ctorch/torch_tensor.h"
#include "common.h"


void torch_tensor_delete(TorchTensor obj) {
    auto value = static_cast<torch::Tensor *>(obj);
    delete value;
}

size_t
torch_tensor_parse_to_bbox(TorchTensor obj, float confidence_threshold, int max_result_size, TensorResultBox **output,
                           TorchStatus *status) {
    auto detections = static_cast<torch::Tensor *>(obj);
    torch_reset_status(status);
    if (output == nullptr) {
        return 0;
    }

    //detections:{1,25200, 85}  85=num_class+attr_size
    // 85 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf, 5~84: class conf
    constexpr int item_attr_size = 5;
    constexpr int object_confidence_idx = 4;
    auto batch_size = detections->size(0);
    try {
        if (batch_size != 1) {
            throw std::runtime_error("batch_size is not 1");
        }

        auto num_bbox_confidence_class_idx = detections->size(2);
        //auto num_classes = num_bbox_confidence_class_idx - item_attr_size;
        at::Tensor candidate_object_mask = detections->select(-1, object_confidence_idx).gt(
                confidence_threshold).unsqueeze(-1);
        at::Tensor candidate_object_tensor = torch::masked_select(detections[0], candidate_object_mask[0]).view(
                {-1, num_bbox_confidence_class_idx});
        if (candidate_object_tensor.size(0) == 0) {
            return 0;
        }

        at::Tensor class_score_tensor = candidate_object_tensor.slice(-1, 4, item_attr_size) *
                                        candidate_object_tensor.slice(-1, item_attr_size);

        auto max_class_score_tuple = torch::max(class_score_tensor, -1);
        // class score
        auto max_conf_score = std::get<0>(max_class_score_tuple).to(torch::kFloat).unsqueeze(1);
        // index
        auto max_conf_index = std::get<1>(max_class_score_tuple).to(torch::kFloat).unsqueeze(1);

        candidate_object_tensor = torch::cat({candidate_object_tensor.slice(1, 0, 4), max_conf_score, max_conf_index},
                                             1);

        auto len = candidate_object_tensor.size(0);
        if (max_result_size > 0 && len > max_result_size) {
            //sort by confidence and remove excess boxes
            auto topIdx = candidate_object_tensor.select(1, 4).argsort(0, true).slice(0, 0, max_result_size);
            candidate_object_tensor = candidate_object_tensor.index_select(0, topIdx);
            len = candidate_object_tensor.size(0);
        }

        auto result_cpu = candidate_object_tensor.cpu();
        auto result_bbox_tensor_accessor = result_cpu.accessor<float, 2>();

        auto data = (TensorResultBox *) malloc(sizeof(TensorResultBox) * len);
        for (int i = 0; i < len; ++i) {
            data[i] = {
                    result_bbox_tensor_accessor[i][0],
                    result_bbox_tensor_accessor[i][1],
                    result_bbox_tensor_accessor[i][2],
                    result_bbox_tensor_accessor[i][3],
                    result_bbox_tensor_accessor[i][4],
                    int(result_bbox_tensor_accessor[i][5])
            };
        }
        *output = data;
        return len;
    } catch (std::exception &e) {
        torch_set_status(status, e);
        return -1;
    }
}
