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

#include "ctorch/torch_interpreter_value.h"
#include "common.h"

void torch_ivalue_delete(TorchIValue obj) {
    auto value = static_cast<torch::jit::IValue *>(obj);
    delete value;
}


bool torch_ivalue_is_tuple(TorchIValue obj) {
    auto value = static_cast<torch::jit::IValue *>(obj);
    return value->isTuple();
}


bool torch_ivalue_is_tensor(TorchIValue obj) {
    auto value = static_cast<torch::jit::IValue *>(obj);
    return value->isTensor();
}

TorchTensor torch_ivalue_to_tensor(TorchIValue obj) {
    auto value = static_cast<torch::jit::IValue *>(obj);
    return new torch::Tensor(value->toTensor());
}

size_t torch_ivalue_length(TorchIValue obj) {
    auto value = static_cast<torch::jit::IValue *>(obj);

    if (value->isTuple()) return value->toTuple()->elements().size();
    else if (value->isIntList()) return value->toIntList().size();
    else if (value->isDoubleList()) return value->toDoubleList().size();
    else if (value->isBoolList()) return value->toBoolList().size();
    else if (value->isString()) return value->toStringRef().size();
    else if (value->isTensorList()) return value->toTensorList().size();
    else if (value->isList()) return value->toList().size();
    else if (value->isGenericDict()) return value->toGenericDict().size();

    return -1;
}

void torch_ivalue_to_tuple(TorchIValue obj, TorchIValue *outputs, size_t output_size) {
    auto value = static_cast<torch::jit::IValue *>(obj);

    auto vec = value->toTuple()->elements();
    if (vec.size() != output_size) {
        //throw std::invalid_argument("unexpected tuple size");
    }
    for (int i = 0; i < output_size; ++i) {
        //outputs[i] = new torch::jit::IValue(vec[i]);
    }
}
