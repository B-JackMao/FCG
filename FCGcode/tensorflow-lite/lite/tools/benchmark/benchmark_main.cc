
// /* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

// #include <iostream>

// #include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
// #include "tensorflow/lite/tools/logging.h"

// namespace tflite {
// namespace benchmark {

// int Main(int argc, char** argv) {
//   TFLITE_LOG(INFO) << "STARTING!";
//   BenchmarkTfLiteModel benchmark;
//   if (benchmark.Run(argc, argv) != kTfLiteOk) {
//     TFLITE_LOG(ERROR) << "Benchmarking failed.";
//     return EXIT_FAILURE;
//   }
//   return EXIT_SUCCESS;
// }
// }  // namespace benchmark
// }  // namespace tflite

// int main(int argc, char** argv) { return tflite::benchmark::Main(argc, argv); }

#include <thread>
#include <sched.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <jpeglib.h>
#include <future>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <queue>
#include <functional>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "ctpl_stl.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include "jpegint.h"
struct band_info {
  short *dct;
  unsigned int dct_h;
  unsigned int dct_w;
  unsigned int dct_b;
};

std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::Interpreter> interpreter2;
std::mutex mtx1;
std::mutex mtx2;
std::mutex mtx3;
std::mutex mtx4;
std::condition_variable cv;
std::queue<std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *>> task_part_queue[4]; // Task queue for each part
std::unordered_map<int, int> task_state; 
bool done = false;
int end_task1 = 2;
int end_task2 = 3;
int end_task2_1 = 2;
int remaining_tasks = 0;
int flag1 = 15000;
int flag2 = 28000;

void unpack_band_info(band_info* band, short **band_dct, int *band_dct_h,
                      int *band_dct_w, int *band_dct_b) {
  *band_dct = band->dct;
  *band_dct_h = band->dct_h;
  *band_dct_w = band->dct_w;
  *band_dct_b = band->dct_b;
}

void dummy_dct_coefficients(band_info *band) {
  band->dct_h = 0;
  band->dct_w = 0;
  band->dct_b = 0;
  band->dct = new short[0];
}
void read_dct_coefficients(jpeg_decompress_struct *srcinfo,
                           short *inputBuffer, int compNum,
                           band_info *band, bool normalized,unsigned int totalBlock,int flag) {
  if (compNum >= srcinfo->num_components) {

    band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2; 
    band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;
    band->dct_b = DCTSIZE2;
    long nb_elements =
        (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);/
    band->dct = new short[nb_elements];
    std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
    return;
  }


  long nb_elements =
      (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);


  int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
  const unsigned short* quantval = normalized ? srcinfo->quant_tbl_ptrs[quant_idx]->quantval : nullptr;
  short unscale = 1;

  short *current_dct_coeff = band->dct;
  short *current_dct_coeff2 = band->dct+(nb_elements/2);
  if(compNum==0){
      if(flag==0){
      for(int i =0; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
                if (normalized) {
                  unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
                }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;

        }
      i++;
      short* src2 = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
              if (normalized) {
                unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
              }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
        }
      i=i+5;
    }
      }
    else{
    for(int i =2; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
        if (normalized) {
          unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
        }
        *current_dct_coeff2=src[j]*unscale;
        current_dct_coeff2++;
        }
      i++;
      short* src2 = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
        if (normalized) {
          unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
        }
        *current_dct_coeff2=src[j]*unscale;
        current_dct_coeff2++;
        }
      i=i+5;
    }
    }
  }
  else{
      for(int i =compNum+3; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
        for (unsigned int j =0 ;j<64;j++){
              if (normalized) {
                unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
              }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
        }
      i=i+6;
    }
  }
}

void add_error_handler(jpeg_error_mgr *jerr) {
  jerr->error_exit = [](j_common_ptr cinfo) {
    char pszErr[1024];
    (cinfo->err->format_message)(cinfo, pszErr);
    fprintf(stderr, "Error: %s\n", pszErr);
    exit(EXIT_FAILURE);
  };
}


void transcode(jpeg_decompress_struct *srcinfo, unsigned char **outbuffer) {
  static bool warning_emitted = false;
  if (!warning_emitted) {
    fprintf(stderr, "WARNING: Non-standard JPEG image encountered, transcoding "
                    "to H2_V2 which may negatively impact performance. This is "
                    "the only time this warning will be shown.\n");
    warning_emitted = true;
  }

  // start decompress
  (void)jpeg_start_decompress(srcinfo);

  // create the compression structure
  jpeg_compress_struct dstinfo;
  jpeg_error_mgr jerr;
  dstinfo.err = jpeg_std_error(&jerr);
  add_error_handler(&jerr);
  jpeg_create_compress(&dstinfo);

  size_t outlen = 0;
  jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

  dstinfo.image_width = srcinfo->image_width;
  dstinfo.image_height = srcinfo->image_height;
  dstinfo.input_components = srcinfo->output_components;
  dstinfo.in_color_space = srcinfo->out_color_space;

  jpeg_set_defaults(&dstinfo);
  jpeg_set_quality(&dstinfo, 100, TRUE);
  jpeg_start_compress(&dstinfo, TRUE);


  unsigned char *line_buffer =
      new unsigned char[srcinfo->output_width * srcinfo->output_components];
  while (srcinfo->output_scanline < srcinfo->output_height) {
    jpeg_read_scanlines(srcinfo, &line_buffer, 1);
    (void)jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
  }
  delete[] line_buffer;
  jpeg_finish_compress(&dstinfo);
  jpeg_destroy_compress(&dstinfo);

  // re-create decompress
  jpeg_destroy_decompress(srcinfo);
  jpeg_create_decompress(srcinfo);
  jpeg_mem_src(srcinfo, *outbuffer, outlen);
  (void)jpeg_read_header(srcinfo, TRUE);
}

bool is_grayscale(jpeg_decompress_struct *srcinfo) {
  return srcinfo->num_components == 1;
}

bool is_h2_v2(jpeg_decompress_struct *srcinfo) {
  return (srcinfo->comp_info[0].h_samp_factor == 2) &&
         (srcinfo->comp_info[0].v_samp_factor == 2) &&
         (srcinfo->comp_info[1].h_samp_factor == 1) &&
         (srcinfo->comp_info[1].v_samp_factor == 1) &&
         (srcinfo->comp_info[2].h_samp_factor == 1) &&
         (srcinfo->comp_info[2].v_samp_factor == 1);
}

struct thread_data {
    int start;
    int end;
    j_decompress_ptr info;
    JBLOCKROW* mcu_buffer;
    JBLOCKROW block_buffer;
    int thread;
    int line;
};
struct result_data {
    int num;
    j_decompress_ptr info;
    short * inputBuffer;
    int thread;
    int normalized;
    unsigned int totalBlock;
    band_info * band;
    int flag;
};
struct result_data2 {
    j_decompress_ptr info;
    short * inputBuffer;
    int thread;
    int normalized;
    unsigned int totalBlock;
    band_info * band;
    band_info * band2;
    int flag;
};


void* process_mcus(void* args) {

    struct thread_data* data = (struct thread_data*)args;

    pid_t tid = syscall(SYS_gettid);


    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread, &cpuset); 
    sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

    for (int i = data->start; i < data->end; ++i) {
        data->info->line[data->line] = i;
        for (int j = 0; j < data->info->blocks_in_MCU; ++j) {
            data->mcu_buffer[j] = data->block_buffer + data->info->blocks_in_MCU * i + j;
        }
        if (FALSE == (*data->info->entropy->decode_mcu_mult)(data->info, data->mcu_buffer, data->line)) {
            break;
        }
    }
    return NULL;
}

void* process_result(void* args) {

    struct result_data* data = (struct result_data*)args;

    pid_t tid = syscall(SYS_gettid);


    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread, &cpuset); 
    sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

    read_dct_coefficients(data->info, data->inputBuffer, data->num, data->band, data->normalized,data->totalBlock,data->flag);
    return NULL;
}
void* process_result2(void* args) {

    struct result_data2* data = (struct result_data2*)args;

    pid_t tid = syscall(SYS_gettid);

  
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread, &cpuset); 
    sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

    read_dct_coefficients(data->info, data->inputBuffer, 1, data->band, data->normalized,data->totalBlock,data->flag);
    read_dct_coefficients(data->info, data->inputBuffer, 2, data->band2, data->normalized,data->totalBlock,data->flag);
    return NULL;
}
void read_dct_coefficients_from_srcinfo_mult(jpeg_decompress_struct *srcinfo,
                                        bool normalized, int channels,
                                        band_info *band1, band_info *band2,
                                        band_info *band3,unsigned char *buffer){
  unsigned int totalMcuNum = srcinfo->MCU_rows_in_scan * srcinfo->MCUs_per_row;
  unsigned int totalBlock = srcinfo->blocks_in_MCU * totalMcuNum;
    short *inputBuffer = (short*)malloc(totalBlock * sizeof(JBLOCK));
  JBLOCK **mcu_buffer = (JBLOCK**)malloc(sizeof(JBLOCK*) * srcinfo->blocks_in_MCU);
  JBLOCK *block_buffer = (JBLOCK*)inputBuffer;
  pthread_t thread1, thread2,thread3,thread4;
  struct thread_data t1_data = {0, flag1 , srcinfo, mcu_buffer, block_buffer,7,1};
  struct thread_data t2_data = {flag1, flag2, srcinfo, mcu_buffer, block_buffer,6,2};
  struct thread_data t3_data = {flag2, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,5,3};
  //struct thread_data t4_data = {31900, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,4,4};
  srcinfo->unread_marker = 0;
  srcinfo->unread_marker2 = 0;
  srcinfo->unread_marker3 = 0;
  srcinfo->now_b1_id=1;
  srcinfo->now_b2_id=srcinfo->huff_scan.buffer_id[flag1]-1;//di er ge xian cheng kai shi de buffer id;
  srcinfo->now_b3_id=srcinfo->huff_scan.buffer_id[flag2]-1;

  (*srcinfo->src->retrun_file_start) (srcinfo);

  pthread_create(&thread1, NULL, process_mcus, &t1_data);

  pthread_create(&thread2, NULL, process_mcus, &t2_data);

  pthread_create(&thread3, NULL, process_mcus, &t3_data);



  pthread_join(thread1, NULL);

  pthread_join(thread2, NULL);

  pthread_join(thread3, NULL);
      //usleep(50000);
  pthread_join(thread4, NULL);
    band1->dct_h = srcinfo->comp_info[0].height_in_blocks;
  band1->dct_w = srcinfo->comp_info[0].width_in_blocks;
  band1->dct_b = DCTSIZE2;
  long nb_elements1 =
      (long)(band1->dct_h) * (long)(band1->dct_w) * (long)(band1->dct_b);
  band1->dct = new short[nb_elements1];
      band2->dct_h = srcinfo->comp_info[1].height_in_blocks;
  band2->dct_w = srcinfo->comp_info[1].width_in_blocks;
  band2->dct_b = DCTSIZE2;
  long nb_elements2 =
      (long)(band2->dct_h) * (long)(band2->dct_w) * (long)(band2->dct_b);
  band2->dct = new short[nb_elements2];
      band3->dct_h = srcinfo->comp_info[2].height_in_blocks;
  band3->dct_w = srcinfo->comp_info[2].width_in_blocks;
  band3->dct_b = DCTSIZE2;
  long nb_elements3 =
      (long)(band3->dct_h) * (long)(band3->dct_w) * (long)(band3->dct_b);
  band3->dct = new short[nb_elements3];
  auto start = std::chrono::high_resolution_clock::now();
  struct result_data r1 = {0,srcinfo,inputBuffer,7,normalized,totalBlock,band1,0};
  struct result_data r3 = {0,srcinfo,inputBuffer,6,normalized,totalBlock,band1,1};
  //struct result_data r2 = {1,srcinfo,inputBuffer,6,normalized,totalBlock,band2};
  //struct result_data r3 = {2,srcinfo,inputBuffer,5,normalized,totalBlock,band3};
  struct result_data2 r2 = {srcinfo,inputBuffer,5,normalized,totalBlock,band2,band3,0};

  pthread_create(&thread1, NULL, process_result, &r1);

  pthread_create(&thread3, NULL, process_result, &r3);

  pthread_create(&thread2, NULL, process_result2, &r2);

  pthread_join(thread1, NULL);

  pthread_join(thread2, NULL);
  pthread_join(thread3, NULL);

  if (buffer != nullptr) {
    delete[] buffer;
  }
}
void read_dct_coefficients_from_file_scan(char *filename,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
  
  if ((infile = fopen(filename, "rb")) == nullptr) {
    fprintf(stderr, "ERROR: can't open %s\n", filename);
    return;
  }

  jpeg_error_mgr jerr;
  srcinfo->err = jpeg_std_error(&jerr);
  add_error_handler(&jerr);
  jpeg_create_decompress(srcinfo);
  jpeg_stdio_src(srcinfo, infile);
  
  jpeg_read_header(srcinfo, TRUE);
  jpeg_start_decompress(srcinfo);

  if (!is_grayscale(srcinfo) && !is_h2_v2(srcinfo)) {
    transcode(srcinfo, &buffer);
  }

  jpeg_read_coefficients_scan(srcinfo);

}
void read_dct_coefficients_from_file_mult(char *filename, bool normalized,
                                      int channels, band_info *band1,
                                      band_info *band2, band_info *band3,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
  read_dct_coefficients_from_srcinfo_mult(srcinfo,normalized,channels,band1,band2,band3,buffer);
  jpeg_abort_decompress(srcinfo);
  jpeg_destroy_decompress(srcinfo);

}


short* resizeShortData(const short* input_data, int original_height, int original_width, int channels,
                       int new_height, int new_width) {

    short* resized_data = new short[new_height * new_width * channels];


    float scale_h = static_cast<float>(original_height - 1) / (new_height - 1);
    float scale_w = static_cast<float>(original_width - 1) / (new_width - 1);


    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float gy = y * scale_h;
            float gx = x * scale_w;

            int gxi = static_cast<int>(gx);
            int gyi = static_cast<int>(gy);

            for (int c = 0; c < channels; ++c) {
                float c00 = input_data[(gyi * original_width + gxi) * channels + c];
                float c10 = input_data[(gyi * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];
                float c01 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + gxi) * channels + c];
                float c11 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];

                float cx0 = c00 + (c10 - c00) * (gx - gxi);
                float cx1 = c01 + (c11 - c01) * (gx - gxi);
                float cxy = cx0 + (cx1 - cx0) * (gy - gyi);

                resized_data[(y * new_width + x) * channels + c] = static_cast<short>(std::round(cxy));
            }
        }
    }

    return resized_data;
}

std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
    std::vector<float> batchedImage(1 * height * width * channels);
    std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
    return batchedImage;
}
std::vector<float> convertToInt8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
    std::vector<float> int8_vector;
    long nb_elements = static_cast<long>(dct_h) * static_cast<long>(dct_w) * static_cast<long>(dct_b);
    int8_vector.reserve(nb_elements);
    for (long i = 0; i < nb_elements; ++i) {
        int8_vector.push_back(static_cast<float>(band_dct[i]));
    }
    return int8_vector;
}

const int height = 14;
const int width = 14;
const int depth1 = 64;  
const int depth2 = 64;  
std::vector<float> mergeInputs(const std::vector<float>& input1, const std::vector<float>& input2) {

    if (input1.size() != height * width * depth1 || input2.size() != height * width * depth2) {
        std::cerr << "no match!" << std::endl;
        return {};
    }


    std::vector<float> merged_input(height * width * (depth1 + depth2));


    for (int i = 0; i < height * width; ++i) {
        
        for (int d1 = 0; d1 < depth1; ++d1) {
            merged_input[i * (depth1 + depth2) + d1] = input1[i * depth1 + d1];
        }
        
        for (int d2 = 0; d2 < depth2; ++d2) {
            merged_input[i * (depth1 + depth2) + depth1 + d2] = input2[i * depth2 + d2];
        }
    }

    return merged_input;
}
void inferenceTask(short* band1_dct, int band1_dct_h, int band1_dct_w, int band1_dct_b,
                         short* band2_dct, int band2_dct_h, int band2_dct_w, int band2_dct_b,
                         short* band3_dct, int band3_dct_h, int band3_dct_w, int band3_dct_b,int id) {

    int int8_min = -128;
    int int8_max = 127;
    int min_val = -1024;
    int max_val = 1016;
    int target_size1=28;
    int target_size2=14;
    //usleep(50000);
    short* resized_data1 = resizeShortData(band1_dct,band1_dct_h, band1_dct_w, band1_dct_b, target_size1, target_size1);

    short* resized_data2 = resizeShortData(band2_dct,band2_dct_h, band2_dct_w, band2_dct_b, target_size2, target_size2);

    short* resized_data3 = resizeShortData(band3_dct,band3_dct_h, band3_dct_w, band3_dct_b, target_size2, target_size2);

    std::vector<float> input_data1(target_size1 * target_size1 * band1_dct_b);
    std::vector<float> input_data2(target_size2 * target_size2 * band2_dct_b);
    std::vector<float> input_data3(target_size2 * target_size2 * band3_dct_b);
    input_data1 = convertToInt8Vector(resized_data1,target_size1, target_size1, band1_dct_b);
    input_data2 = convertToInt8Vector(resized_data2,target_size2, target_size2, band2_dct_b);
    input_data3 = convertToInt8Vector(resized_data3,target_size2, target_size2, band3_dct_b);

    std::vector<float> input1_batch = addBatchDimension(input_data1,target_size1, target_size1, band1_dct_b);
    std::vector<float> input2_batch = addBatchDimension(input_data2,target_size2, target_size2, band2_dct_b);
    std::vector<float> input3_batch = addBatchDimension(input_data3,target_size2, target_size2, band3_dct_b);
    std::vector<float> merged_input = mergeInputs(input2_batch, input3_batch);

    int input_index1 = interpreter->inputs()[0];
    int input_index2 = interpreter->inputs()[1];



    TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
    TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
 
    std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
    std::memcpy(tensor2->data.f, merged_input.data(), merged_input.size());

    interpreter->Invoke();

}

// Worker function for part 1 using read_dct_coefficients_from_file
void worker_decode(int cpu_id) {
    while (true) {
        std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
        {
            std::unique_lock<std::mutex> lock(mtx1);
            cv.wait(lock, [] { return (!task_part_queue[0].empty() && end_task1 && end_task2) || done; });
            end_task1--;
            end_task2--;
            if (done && task_part_queue[0].empty())
                break;

            task = task_part_queue[0].front();
            task_part_queue[0].pop();
                              
        }

        // Extract parameters from tuple
        char* filename = std::get<0>(task);
        bool normalized = std::get<1>(task);
        int channels = std::get<2>(task);
        short **band1_dct = std::get<3>(task);
        int *band1_dct_h = std::get<4>(task);
        int *band1_dct_w = std::get<5>(task);
        int *band1_dct_b = std::get<6>(task);
        short **band2_dct = std::get<7>(task);
        int *band2_dct_h = std::get<8>(task);
        int *band2_dct_w = std::get<9>(task);
        int *band2_dct_b = std::get<10>(task);
        short **band3_dct = std::get<11>(task);
        int *band3_dct_h = std::get<12>(task);
        int *band3_dct_w = std::get<13>(task);
        int *band3_dct_b = std::get<14>(task);
        int id = std::get<15>(task);
        FILE *infile = std::get<16>(task);
        band_info *band1 = std::get<17>(task);
        band_info *band2 = std::get<18>(task);
        band_info *band3 = std::get<19>(task);
        jpeg_decompress_struct *srcinfo = std::get<20>(task); 
        unsigned char *buffer = std::get<21>(task);
 	    cpu_set_t mask;
      CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
      CPU_SET(4, &mask);
      if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
       }
        auto start_1 = std::chrono::high_resolution_clock::now();
        // Call the DCT read function
        read_dct_coefficients_from_file_scan(filename,infile,srcinfo,buffer);

        // Update task state and notify other threads
        {
            std::lock_guard<std::mutex> lock(mtx1);
            task_part_queue[1].push(task);  // Queue forpart 2
        }
        cv.notify_all();  // Notify all worker threads
    }
}


// Worker function for part 2
void worker_inference1(int cpu_id) {

    while (true) {
        std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
        {   
            std::unique_lock<std::mutex> lock(mtx2);
            cv.wait(lock, [] { return (!task_part_queue[1].empty() && end_task2_1) || done; });
            end_task2_1--;
            if (done && task_part_queue[1].empty())
                break;

            task = task_part_queue[1].front();
            task_part_queue[1].pop();
        }

        char* filename = std::get<0>(task);
        bool normalized = std::get<1>(task);
        int channels = std::get<2>(task);
        short **band1_dct = std::get<3>(task);
        int *band1_dct_h = std::get<4>(task);
        int *band1_dct_w = std::get<5>(task);
        int *band1_dct_b = std::get<6>(task);
        short **band2_dct = std::get<7>(task);
        int *band2_dct_h = std::get<8>(task);
        int *band2_dct_w = std::get<9>(task);
        int *band2_dct_b = std::get<10>(task);
        short **band3_dct = std::get<11>(task);
        int *band3_dct_h = std::get<12>(task);
        int *band3_dct_w = std::get<13>(task);
        int *band3_dct_b = std::get<14>(task);
        int id = std::get<15>(task);
        FILE *infile = std::get<16>(task);
        band_info *band1 = std::get<17>(task);
        band_info *band2 = std::get<18>(task);
        band_info *band3 = std::get<19>(task);
        jpeg_decompress_struct *srcinfo = std::get<20>(task); 
        unsigned char *buffer = std::get<21>(task);
         auto start_2 = std::chrono::high_resolution_clock::now();
        read_dct_coefficients_from_file_mult(filename,normalized,channels,band1,band2,band3,infile,srcinfo,buffer);
        unpack_band_info(band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
        unpack_band_info(band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
        unpack_band_info(band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
        auto end_1 = std::chrono::high_resolution_clock::now();
//std::cout<<"huffman end---------------"<<std::endl;
        inferenceTask(*band1_dct, *band1_dct_h, *band1_dct_w, *band1_dct_b,*band2_dct, *band2_dct_h, *band2_dct_w, *band2_dct_b,*band3_dct, *band3_dct_h, *band3_dct_w, *band3_dct_b,id);

        auto end_2 = std::chrono::high_resolution_clock::now();
        std::cout<<"stage2:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count()<<std::endl;
        std::cout<<"stageinvoke:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - end_1).count()<<std::endl;
        // Update task state and notify other threads
        {
            std::lock_guard<std::mutex> lock(mtx2);
            task_part_queue[2].push(task); // Queue for part 3
            end_task1++;
            
        }
        cv.notify_all();  // Notify all worker threads
    }
}

// Worker function for part 3
void worker_inference2(int cpu_id) {
    while (true) {
        std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
        {
            std::unique_lock<std::mutex> lock(mtx3);
            cv.wait(lock, [] { return !task_part_queue[2].empty() || done; });
            if (done && task_part_queue[2].empty())
                break;

            task = task_part_queue[2].front();
            task_part_queue[2].pop();
        }
        int id = std::get<15>(task);
        inferenceTask2(id);
        {
            std::lock_guard<std::mutex> lock(mtx3);
            end_task2++;
            end_task2_1++;
            --remaining_tasks;
        }

        cv.notify_all();
        
    }
}

int main(int argc, char* argv[]){
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
        return 1;
    }
      //cpu_set_t mask;
      //CPU_ZERO(&mask);   
      //CPU_SET(, &mask);
      //if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
      // }
    flag1 = std::stoi(argv[1]);
    flag2 = std::stoi(argv[2]);
    std::string modelPath = argv[4];

    // 2. Initialize TensorFlow Lite interpreter
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    tflite::ops::builtin::BuiltinOpResolver* resolver = nullptr;
    resolver = new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
    tflite::InterpreterBuilder builder(*model, *resolver);
    builder(&interpreter);

    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();

    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply GPU delegate." << std::endl;
        TfLiteGpuDelegateV2Delete(delegate);
        return -1;
    }

    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return -1;
    }

    // 3. Allocate tensors and set inputs
    interpreter->AllocateTensors();
    interpreter->SetNumThreads(3);
        char* files[8] = {
        "83.jpeg", "84.jpeg", "85.jpeg",
        "86.jpeg", "87.jpeg", "88.jpeg",
        "89.jpeg", "90.jpeg"
    };

    // Start worker threads for each part
    std::vector<std::thread> workers;
    workers.emplace_back(worker_decode, 1);    // CPU 1 for decode
    workers.emplace_back(worker_inference1, 2); // CPU 2 for inferenceTask
    workers.emplace_back(worker_inference2, 3); // CPU 3 for inferenceTask2
    remaining_tasks = std::stoi(argv[3]);
    int size = 1000*1000*64;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < remaining_tasks; i++) {// jiang suo you ren wu jia ru dao dui lei zhong 
        char* filename = files[i%8];
        short* band1_dct = new short[size];
        int* band1_dct_h = new int(0);
        int* band1_dct_w = new int(0);
        int* band1_dct_b = new int(0);
        short* band2_dct = new short[size];
        int* band2_dct_h = new int(0);
        int* band2_dct_w = new int(0);
        int* band2_dct_b = new int(0);
        short* band3_dct = new short[size];
        int* band3_dct_h = new int(0);
        int* band3_dct_w = new int(0);
        int* band3_dct_b = new int(0);
        bool normalized = true;  // Example value
        int channels = 3;        // Example value
        FILE* infile = nullptr;
        band_info* band1 = new band_info();
        band_info* band2 = new band_info();
        band_info* band3 = new band_info();
        jpeg_decompress_struct* srcinfo = new jpeg_decompress_struct();
        unsigned char* buffer = nullptr;
        {
          std::lock_guard<std::mutex> lock(mtx1);
                    task_part_queue[0].push(std::make_tuple(filename, normalized, channels, 
                                                static_cast<short**>(&band1_dct), band1_dct_h, band1_dct_w, band1_dct_b,
                                                static_cast<short**>(&band2_dct), band2_dct_h, band2_dct_w, band2_dct_b,
                                                static_cast<short**>(&band3_dct), band3_dct_h, band3_dct_w, band3_dct_b,
                                                i, infile, band1, band2, band3, srcinfo, buffer));
    }
        cv.notify_all();  // Notify worker threads
    }
    // Signal threads to stop when done
    {
        //std::lock_guard<std::mutex> lock(mtx);
        std::unique_lock<std::mutex> lock(mtx4);
        cv.wait(lock, [] {
             return remaining_tasks == 0;
        });
        done = true;
    }
    auto end = std::chrono::high_resolution_clock::now();
    cv.notify_all();
    for (auto& worker : workers) {
        worker.join();
    }
  auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout<<"all:"<<duration_invoke.count()<<std::endl;
	return 0;
}