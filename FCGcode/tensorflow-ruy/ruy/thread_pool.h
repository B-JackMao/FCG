/* Copyright 2019 Google LLC. All Rights Reserved.

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

// This file is a fork of gemmlowp's multi_thread_gemm.h, under Apache 2.0
// license.

#ifndef RUY_RUY_THREAD_POOL_H_
#define RUY_RUY_THREAD_POOL_H_

#include <vector>
#include <iostream>
#include "ruy/blocking_counter.h"
#include "ruy/time.h"

namespace ruy {


// A workload for a thread.
struct Task {
  virtual ~Task() {}
  virtual void Run() = 0;
};

class Thread;

// A simple pool of threads, that only allows the very
// specific parallelization pattern that we use here:
// One thread, which we call the 'main thread', calls Execute, distributing
// a Task each to N threads, being N-1 'worker threads' and the main thread
// itself. After the main thread has completed its own Task, it waits for
// the worker threads to have all completed. That is the only synchronization
// performed by this ThreadPool.
//
// In particular, there is a naive 1:1 mapping of Tasks to threads.
// This ThreadPool considers it outside of its own scope to try to work
// with fewer threads than there are Tasks. The idea is that such N:M mappings
// of tasks to threads can be implemented as a higher-level feature on top of
// the present low-level 1:1 threadpool. For example, a user might have a
// Task subclass referencing a shared atomic counter indexing into a vector of
// finer-granularity subtasks. Different threads would then concurrently
// increment this atomic counter, getting each their own subtasks to work on.
// That approach is the one used in ruy's multi-thread matrix multiplication
// implementation --- see ruy's TrMulTask.
class ThreadPool {
 public:
  ThreadPool() {}

  ~ThreadPool();
void setCurrentThreadAffinityMask1(int cpu1, int cpu2 = 0, int cpu3 = 0);
  // Executes task_count tasks on task_count threads.
  // Grows the threadpool as needed to have at least (task_count-1) threads.
  // The 0-th task is run on the thread on which Execute is called: that
  // is by definition what we call the "main thread". Synchronization of all
  // threads is performed before this function returns.
  //
  // As explained in the class comment, there is a 1:1 mapping of tasks to
  // threads. If you need something smarter than that, for instance if you
  // want to run an unbounded number of tasks on a bounded number of threads,
  // then you need something higher-level than this ThreadPool, that can
  // be layered on top of it by appropriately subclassing Tasks.
  //
  // TaskType must be a subclass of ruy::Task. That is implicitly guarded by
  // the static_cast in this inline implementation.
  template <typename TaskType>
  void Execute(int task_count, TaskType* tasks) {
    //std::cout<<"th:"<<tasks->th<<std::endl;
    ExecuteImpl(task_count, sizeof(TaskType), static_cast<Task*>(tasks));
  }

  void set_spin_milliseconds(float milliseconds) {
    spin_duration_ = DurationFromMilliseconds(milliseconds);
  }

  float spin_milliseconds() const {
    return ToFloatMilliseconds(spin_duration_);
  }

 private:
  // Ensures that the pool has at least the given count of threads.
  // If any new thread has to be created, this function waits for it to
  // be ready.

  void CreateThreads(int threads_count);

  // Non-templatized implementation of the public Execute method.
  // See the inline implementation of Execute for how this is used.
  void ExecuteImpl(int task_count, int stride, Task* tasks);

  // copy construction disallowed
  ThreadPool(const ThreadPool&) = delete;

  // The threads in this pool. They are owned by the pool:
  // the pool creates threads and destroys them in its destructor.
  std::vector<Thread*> threads_;
  int flag_change_core = 0;

  // The BlockingCounter used to wait for the threads.
  BlockingCounter counter_to_decrement_when_ready_;

  // This value was empirically derived with some microbenchmark, we don't have
  // high confidence in it.
  //
  // That this value means that we may be sleeping substantially longer
  // than a scheduler timeslice's duration is not necessarily surprising. The
  // idea is to pick up quickly new work after having finished the previous
  // workload. When it's new work within the same GEMM as the previous work, the
  // time interval that we might be busy-waiting is very small, so for that
  // purpose it would be more than enough to sleep for 1 ms.
  // That is all what we would observe on a GEMM benchmark. However, in a real
  // application, after having finished a GEMM, we might do unrelated work for
  // a little while, then start on a new GEMM. In that case the wait interval
  // may be a little longer. There may also not be another GEMM for a long time,
  // in which case we'll end up passively waiting below.
  Duration spin_duration_ = DurationFromMilliseconds(2);
};
extern void set_core(int core);
}  // namespace ruy

#endif  // RUY_RUY_THREAD_POOL_H_
