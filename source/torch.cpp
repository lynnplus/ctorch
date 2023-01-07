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

#include <ctorch/torch_core.h>
#include <exception>
#include <string>


void torch_status_clear(TorchStatus *status) {
    if (status == nullptr) {
        return;
    }
    if (status->msg != nullptr) {
        free(status->msg);
        status->msg = nullptr;
    }
    status->code = 0;
}

void torch_reset_status(TorchStatus *status) {
    if (status == nullptr) {
        return;
    }
    status->code = 0;
    status->msg = nullptr;
}

void torch_set_status(TorchStatus *status, std::exception &e, int code) {
    if (status == nullptr || e.what() == nullptr) {
        return;
    }
    status->code = code;
    status->msg = (char *) malloc(std::strlen(e.what()) + 1);
    if (status->msg != nullptr) {
        strcpy(status->msg, e.what());
    }
}