/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

CLCommandQueue::CLCommandQueue(cl_command_queue queue, bool has_ownership)
    : queue_(queue), has_ownership_(has_ownership) {}

CLCommandQueue::CLCommandQueue(CLCommandQueue&& queue)
    : queue_(queue.queue_), has_ownership_(queue.has_ownership_) {
  queue.queue_ = nullptr;
}

CLCommandQueue& CLCommandQueue::operator=(CLCommandQueue&& queue) {
  if (this != &queue) {
    Release();
    std::swap(queue_, queue.queue_);
    has_ownership_ = queue.has_ownership_;
  }
  return *this;
}

CLCommandQueue::~CLCommandQueue() { Release(); }

void CLCommandQueue::Release() {
  if (has_ownership_ && queue_) {
    clReleaseCommandQueue(queue_);
    queue_ = nullptr;
  }
}

absl::Status CLCommandQueue::Dispatch(const CLKernel& kernel,
                                      const int3& work_groups_count,
                                      const int3& work_group_size,
                                      CLEvent* event) {
  std::array<size_t, 3> local;
  std::array<size_t, 3> global;
  for (int i = 0; i < 3; ++i) {
    local[i] = work_group_size[i];
    global[i] = work_groups_count[i] * work_group_size[i];
  }
  cl_event resulting_event;
  const int error_code = clEnqueueNDRangeKernel(
      queue_, kernel.kernel(), 3, nullptr, global.data(), local.data(), 0,
      nullptr, event ? &resulting_event : nullptr);
  if (event) {
    *event = CLEvent(resulting_event);
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clEnqueueNDRangeKernel - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::Dispatch(const CLKernel& kernel,
                                      const int3& work_groups_count,
                                      const int3& work_group_size) {
  return Dispatch(kernel, work_groups_count, work_group_size, nullptr);
}

absl::Status CLCommandQueue::EnqueueEvent(CLEvent* event) {
  cl_event resulting_event;
  const int error_code = clEnqueueMarker(queue_, &resulting_event);
  *event = CLEvent(resulting_event);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to clEnqueueMarker - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueWriteImage(cl_mem memory, int3 region,
                                               const void* data, bool async) {
  const size_t origin[] = {0, 0, 0};
  const size_t r[] = {static_cast<size_t>(region.x),
                      static_cast<size_t>(region.y),
                      static_cast<size_t>(region.z)};
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueWriteImage(queue_, memory, blocking, origin, r, 0,
                                        0, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to upload data to GPU (clEnqueueWriteImage) - ",
                     CLErrorCodeToString(error_code)));
  }

  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueReadImage(cl_mem memory, int3 region,
                                              void* data, bool async) {
  const size_t origin[] = {0, 0, 0};
  const size_t r[] = {static_cast<size_t>(region.x),
                      static_cast<size_t>(region.y),
                      static_cast<size_t>(region.z)};
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueReadImage(queue_, memory, blocking, origin, r, 0,
                                       0, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to read data from GPU (clEnqueueReadImage) - ",
                     CLErrorCodeToString(error_code)));
  }

  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueWriteBuffer(cl_mem memory,
                                                size_t size_in_bytes,
                                                const void* data, bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueWriteBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to upload data to GPU (clEnqueueWriteBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueReadBuffer(cl_mem memory,
                                               size_t size_in_bytes, void* data,
                                               bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueReadBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to read data from GPU (clEnqueueReadBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}
void CL_CALLBACK myCallback(cl_event event, cl_int status, void* user_data) {
  clReleaseEvent(event);
  printf("Event %p completed with status %d\n", event, status);
}
absl::Status CLCommandQueue::EnqueueMapBuffer(cl_mem memory,
                                              size_t size_in_bytes,
                                              const void* data) {
  //std::cout<<"map"<<std::endl;
  
  cl_int error_code;
  cl_event map_event; 
  //std::cout<<"data in map:"<<*(double*)data<<std::endl;
  void* mapped_ptr = clEnqueueMapBuffer(queue_, memory, CL_FALSE, CL_MAP_WRITE, 0, size_in_bytes, 0, nullptr, &map_event, &error_code);// 拷贝数据从映射的指针 
  if (mapped_ptr == nullptr) {
      return absl::InternalError("Failed to map output buffer");
    }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to map data to CPU (clEnqueueMapBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  auto start_copy_to_gpu = std::chrono::high_resolution_clock::now();
  memcpy(mapped_ptr, data, size_in_bytes); // 使用clEnqueueUnmapMemObject来释放映射的指针 
  auto end_copy_to_gpu = std::chrono::high_resolution_clock::now();
  auto duration_copy_to_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_copy_to_gpu - start_copy_to_gpu);
  //std::cout<<"duration_copy_to_gpu: "<<duration_copy_to_gpu.count()<<std::endl;
  cl_event unmap_event; 
  cl_int err=clEnqueueUnmapMemObject(queue_, memory, mapped_ptr, 0, nullptr, &unmap_event); // 等待事件完成 
  if (err != CL_SUCCESS) {
      return absl::InternalError("Failed to unmap output buffer");
    }
  err =clWaitForEvents(1, &unmap_event); // 返回成功状态 
  if (err != CL_SUCCESS) {
      return absl::InternalError("Failed to wait for unmap event");
    }
    //释放映射和解映射的事件
    clReleaseEvent(map_event);
    clReleaseEvent(unmap_event);
    

  return absl::OkStatus(); 
  
  // clRetainEvent(map_event);
  // clRetainEvent(unmap_event);
  // err = clSetEventCallback(map_event, CL_COMPLETE, myCallback, NULL); 
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to set event callback");
  //   }
  // err = clSetEventCallback(unmap_event, CL_COMPLETE, myCallback, NULL);
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to set event callback");
  //   }
  //return absl::OkStatus();
}
// void CL_CALLBACK myCallback(cl_event event, cl_int status, void* user_data) {
//   clReleaseEvent(event);
//   printf("Event %p completed with status %d\n", event, status);
// }
void *unmap_data=nullptr;
void* unmap_mapped_ptr=nullptr;
size_t unmap_size_in_bytes;
absl::Status CLCommandQueue::EnqueueUnMapBuffer(cl_mem memory,
                                              size_t size_in_bytes,
                                              void* data) {

  //std::cout<<"unmap"<<std::endl;

  cl_event map_event;
  cl_int error_code; 
  //std::cout<<"data1 out map:"<<*(float*)data<<std::endl;
  void* mapped_ptr = clEnqueueMapBuffer(queue_, memory, CL_FALSE, CL_MAP_READ, 0, size_in_bytes, 0, nullptr, &map_event, &error_code);// 拷贝数据从映射的指针 
  if (mapped_ptr == nullptr) {
      return absl::InternalError("Failed to map input buffer");
    }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to map data to CPU (clEnqueueMapBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  //err = clSetEventCallback(map_event, CL_COMPLETE, myCallback, NULL); 
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to set event callback");
  //   }
    unmap_data=data;
    unmap_mapped_ptr=mapped_ptr;
    unmap_size_in_bytes=size_in_bytes;
   // 使用clEnqueueUnmapMemObject来释放映射的指针 
  
  // cl_event unmap_event; 
  //  cl_int err =clEnqueueUnmapMemObject(queue_, memory, mapped_ptr, 0, nullptr, &unmap_event); // 等待事件完成 
  //  if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to unmap input buffer");
  //   }
  // clRetainEvent(map_event);
  // clRetainEvent(unmap_event);
  // err = clSetEventCallback(map_event, CL_COMPLETE, myCallback, NULL); 
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to set event callback");
  //   }
  // err = clSetEventCallback(unmap_event, CL_COMPLETE, myCallback, NULL);
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to set event callback");
  //   }  
  //   err =clWaitForEvents(1, &unmap_event); // 返回成功状态 
  // if (err != CL_SUCCESS) {
  //     return absl::InternalError("Failed to wait for unmap event");
  //   }
    //释放映射和解映射的事件
    // clReleaseEvent(map_event);
    // clReleaseEvent(unmap_event);
  return absl::OkStatus(); 
}

absl::Status CLCommandQueue::WaitForCompletion() {
  auto error_code = clFinish(queue_);
  std::memcpy( unmap_data,unmap_mapped_ptr,unmap_size_in_bytes);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clFinish - ", CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

ProfilingCommandQueue::ProfilingCommandQueue(cl_command_queue queue)
    : CLCommandQueue(queue, true) {
  events_.reserve(128);
}

ProfilingCommandQueue::ProfilingCommandQueue(ProfilingCommandQueue&& queue)
    : CLCommandQueue(std::move(queue)),
      events_(std::move(queue.events_)),
      number_of_dispatches_(std::move(queue.number_of_dispatches_)),
      current_label_(std::move(queue.current_label_)) {}

ProfilingCommandQueue& ProfilingCommandQueue::operator=(
    ProfilingCommandQueue&& queue) {
  if (this != &queue) {
    events_ = std::move(queue.events_);
    number_of_dispatches_ = std::move(queue.number_of_dispatches_);
    current_label_ = std::move(queue.current_label_);
    CLCommandQueue::operator=(std::move(queue));
  }
  return *this;
}

void ProfilingCommandQueue::SetEventsLabel(const std::string& name) {
  current_label_ = name;
}

void ProfilingCommandQueue::ResetMeasurements() {
  events_.clear();
  number_of_dispatches_.clear();
}

absl::Status ProfilingCommandQueue::Dispatch(const CLKernel& kernel,
                                             const int3& work_groups_count,
                                             const int3& work_group_size) {
  events_.push_back(CLEvent());
  number_of_dispatches_.push_back(1);
  RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                           work_group_size,
                                           &events_[events_.size() - 1]));
  events_.back().SetName(current_label_);
  return absl::OkStatus();
}

absl::Status ProfilingCommandQueue::DispatchNTimes(
    const CLKernel& kernel, const int3& work_groups_count,
    const int3& work_group_size, int n, int flush_period) {
  number_of_dispatches_.push_back(n);
  if (n == 1) {
    events_.push_back(CLEvent());
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 1]));
    events_.back().SetName(current_label_);
  } else {
    events_.push_back(CLEvent());
    events_.push_back(CLEvent());
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 2]));
    for (int i = 1; i < n - 1; ++i) {
      RETURN_IF_ERROR(
          CLCommandQueue::Dispatch(kernel, work_groups_count, work_group_size));
      if (flush_period && i % flush_period == 0) {
        clFlush(queue_);
      }
    }
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 1]));
    clFlush(queue_);
    events_[events_.size() - 2].SetName(current_label_);
    events_[events_.size() - 1].SetName(current_label_);
  }
  return absl::OkStatus();
}

ProfilingInfo ProfilingCommandQueue::GetProfilingInfo() const {
  ProfilingInfo result;
  result.dispatches.resize(number_of_dispatches_.size());
  int events_counter = 0;
  for (int i = 0; i < number_of_dispatches_.size(); ++i) {
    result.dispatches[i].label = events_[events_counter].GetName();
    if (number_of_dispatches_[i] == 1) {
      result.dispatches[i].duration =
          absl::Nanoseconds(events_[events_counter].GetEventTimeNs());
      events_counter += 1;
    } else {
      result.dispatches[i].duration =
          absl::Nanoseconds(events_[events_counter + 1].GetFinishedTimeNs() -
                            events_[events_counter].GetStartedTimeNs()) /
          number_of_dispatches_[i];
      events_counter += 2;
    }
  }
  return result;
}

absl::Status ProfilingCommandQueue::GetBestWorkGroupIndex(
    const CLKernel& kernel, const GpuInfo& gpu_info,
    const std::vector<int3>& work_groups_count,
    const std::vector<int3>& work_group_sizes, int* index) {
  // Some Adreno 3xx can have wrong numbers for some events
  const bool possible_bug_with_events =
      gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx();
  events_.resize(work_group_sizes.size());
  for (int i = 0; i < work_group_sizes.size(); ++i) {
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count[i],
                                             work_group_sizes[i], &events_[i]));

    // reducing the speed of memory leak on Mali for some kernels
    if (gpu_info.IsMali() && i % 8 == 7) {
      events_[i - 7].Wait();
    }
    if (possible_bug_with_events) {
      // We are trying to increase probability for correct result.
      RETURN_IF_ERROR(WaitForCompletion());
    }
  }

  RETURN_IF_ERROR(WaitForCompletion());

  // To release memory of some kernel pool on Mali.
  if (gpu_info.IsMali()) {
    RETURN_IF_ERROR(kernel.ReInit());
  }

  int minimum_index = 0;
  double minimum_time = std::numeric_limits<double>::max();
  if (possible_bug_with_events) {  // we will try to cut out suspicious results
    double average_time = 0.0;
    int average_samples_count = 0;
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      if (events_[i].GetEventTimeMs() < 100 * 1000) {  // 100 sec
        average_time += events_[i].GetEventTimeMs();
        average_samples_count++;
      }
    }
    average_time /= average_samples_count;
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      double time = events_[i].GetEventTimeMs();
      if (time < minimum_time && time >= 0.1 * average_time) {
        minimum_index = i;
        minimum_time = time;
      }
    }
  } else {
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      double time = events_[i].GetEventTimeMs();
      if (time < minimum_time) {
        minimum_index = i;
        minimum_time = time;
      }
    }
  }

  *index = minimum_index;

  return absl::OkStatus();
}

absl::Status CreateCLCommandQueue(const CLDevice& device,
                                  const CLContext& context,
                                  CLCommandQueue* result) {
  int error_code;
  cl_command_queue queue =
      clCreateCommandQueue(context.context(), device.id(), 0, &error_code);
  if (!queue) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a command queue - ",
                     CLErrorCodeToString(error_code)));
  }
  *result = CLCommandQueue(queue, true);
  return absl::OkStatus();
}

double ProfilingCommandQueue::GetQueueExecutionTimeMs() const {
  const uint64_t start = events_.front().GetStartedTimeNs();
  const uint64_t end = events_.back().GetFinishedTimeNs();
  const uint64_t time_ns = (end - start);

  return static_cast<double>(time_ns) / 1000000.0;
}

double ProfilingCommandQueue::GetSumOfEventsTimeMs() const {
  double sum = 0.0;
  for (int i = 0; i < events_.size(); ++i) {
    sum += events_[i].GetEventTimeMs();
  }
  return sum;
}

absl::Status CreateProfilingCommandQueue(const CLDevice& device,
                                         const CLContext& context,
                                         ProfilingCommandQueue* result) {
  int error_code;
  cl_command_queue queue = clCreateCommandQueue(
      context.context(), device.id(), CL_QUEUE_PROFILING_ENABLE, &error_code);
  if (!queue) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a command queue - ",
                     CLErrorCodeToString(error_code)));
  }

  *result = ProfilingCommandQueue(queue);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
