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

#include "ctorch/torch_module.h"
#include "common.h"


TorchModule torch_module_load(const char *model_path, TorchStatus *status) {
    torch_reset_status(status);
    try {
        auto module = torch::jit::load(model_path);
        return new torch::jit::Module(module);
    } catch (std::exception &e) {
        torch_set_status(status, e);
        return nullptr;
    }
}

void torch_module_delete(TorchModule obj) {
    auto mod = static_cast<torch::jit::Module *>(obj);
    delete mod;
}

int torch_module_to_device(TorchModule obj, TorchDevice *device, bool non_blocking, TorchStatus *status) {
    torch_reset_status(status);
    if (device != nullptr) {
        if (device->deviceType == TorchDeviceType_CPU) {
            device->deviceIndex = 0;
        }
        try {
            auto mod = static_cast<torch::jit::Module *>(obj);
            mod->to(torch_device_from_(device), non_blocking);
            return 0;
        } catch (std::exception &e) {
            torch_set_status(status, e);
            return 1;
        }
    }
    return 0;
}

int torch_module_to_scalar(TorchModule obj, TorchScalarType st, bool non_blocking, TorchStatus *status) {
    torch_reset_status(status);
    try {
        auto mod = static_cast<torch::jit::Module *>(obj);
        mod->to(torch::ScalarType(st), non_blocking);
        return 0;
    } catch (std::exception &e) {
        torch_set_status(status, e);
        return 1;
    }
}


TorchTensor torch_module_forward_by_blob(TorchModule obj, TorchBlob *blob, TorchDevice *blobDevice, bool half) {
    auto mod = static_cast<torch::jit::Module *>(obj);
    auto device = torch_device_from_(blobDevice);
    auto tensor_img = torch::from_blob(blob->data, {blob->batchSize, blob->height, blob->width, blob->channels}).to(
            device);
    if (half) {
        tensor_img = tensor_img.to(torch::kHalf);
    }

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);
    torch::jit::IValue output = mod->forward(inputs);
    auto tensor = output.toTuple()->elements()[0].toTensor();
    return new torch::Tensor(tensor);
}