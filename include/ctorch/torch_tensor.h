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

#ifndef CTORCH_TORCH_TENSOR_H
#define CTORCH_TORCH_TENSOR_H


#ifdef __cplusplus
extern "C" {
#endif


#include "torch_core.h"

typedef struct {
    //bounding box
    float centerX, centerY, width, height;
    float score;
    int class_idx;
} TensorResultBox;

/**
 * parse a tensor to bounding box array(no nms processing)
 * @param obj tensor
 * @param confidence_threshold
 * @param max_result_size maximum number of result boxes return <=0:no limit
 * @param outputs return bounding box array (free by the @torch_tensor_result_box_delete when not needed)
 * @param status result status, when an error occurs (code! =0)
 * @return >0:outputs size ==0:no result or outputs is nil <0:error(for detailed errors, can view status)
 */
CTORCH_PUBLIC size_t
torch_tensor_parse_to_bbox(TorchTensor obj, float confidence_threshold, int max_result_size, TensorResultBox **output,
                           TorchStatus *status);


CTORCH_PUBLIC void torch_tensor_delete(TorchTensor obj);

CTORCH_PUBLIC void torch_tensor_result_box_delete(TensorResultBox *output);

#ifdef __cplusplus
}
#endif

#endif //CTORCH_TORCH_TENSOR_H
