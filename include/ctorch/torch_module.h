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

#ifndef CTORCH_TORCH_MODULE_H
#define CTORCH_TORCH_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "torch_core.h"

/**
 * use torch jit load model(torchscript), use @torch_module_delete destroy
 * @param model_path
 * @param status
 * @return
 */
CTORCH_PUBLIC TorchModule torch_module_load(const char *model_path, TorchStatus *status);
CTORCH_PUBLIC void torch_module_delete(TorchModule obj);


CTORCH_PUBLIC int torch_module_to_device(TorchModule obj, TorchDevice *device, bool non_blocking, TorchStatus *status);
CTORCH_PUBLIC int torch_module_to_scalar(TorchModule obj, TorchScalarType st, bool non_blocking, TorchStatus *status);

CTORCH_PUBLIC TorchIValue
torch_module_forward_by_blob(TorchModule obj, TorchBlob *blob, TorchDevice *blobDevice, bool half);


#ifdef __cplusplus
}
#endif

#endif //CTORCH_TORCH_MODULE_H
