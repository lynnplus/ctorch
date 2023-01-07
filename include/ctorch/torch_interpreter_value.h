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

#ifndef CTORCH_TORCH_INTERPRETER_VALUE_H
#define CTORCH_TORCH_INTERPRETER_VALUE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#include "torch_core.h"

CTORCH_PUBLIC void torch_ivalue_delete(TorchIValue obj);

CTORCH_PUBLIC bool torch_ivalue_is_tuple(TorchIValue obj);
CTORCH_PUBLIC bool torch_ivalue_is_tensor(TorchIValue obj);

CTORCH_PUBLIC size_t torch_ivalue_length(TorchIValue obj);
CTORCH_PUBLIC void torch_ivalue_to_tuple(TorchIValue obj, TorchIValue *outputs, size_t output_size);


CTORCH_PUBLIC TorchTensor torch_ivalue_to_tensor(TorchIValue obj);


#ifdef __cplusplus
}
#endif

#endif //CTORCH_TORCH_INTERPRETER_VALUE_H
