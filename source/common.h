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

#ifndef CTORCH_COMMON_H
#define CTORCH_COMMON_H

#include <torch/script.h>
#include <torch/torch.h>
#include <ctorch/torch_core.h>

inline torch::Device torch_device_from_(TorchDevice *device) {
    if (device == nullptr || device->deviceType == TorchDeviceType_CPU) {
        return {torch::kCPU};
    }
    return {static_cast<torch::DeviceType>(device->deviceType), static_cast<torch::DeviceIndex>(device->deviceIndex)};
}

void torch_set_status(TorchStatus *status, std::exception &e, int code = 1);

void torch_reset_status(TorchStatus *status);


#endif //CTORCH_COMMON_H
