//Copyright (c) 2018 Uber Technologies, Inc.
//
//Licensed under the Uber Non-Commercial License (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at the root directory of this project.
//
//See the License for the specific language governing permissions and
//limitations under the License.
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <stdio.h>
#include <sched.h>
// Has to come after `stdio.h`
#include <jpeglib.h>
#include "jpegint.h"
#include <chrono>
#include "dctfromjpg.h"
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
namespace jpeg2dct {
namespace common {

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
                           band_info *band, bool normalized,unsigned int totalBlock) {
  if (compNum >= srcinfo->num_components) {// 检查通道索引是否合法
    // make an empty component which would be half size of chroma // 创建一个空的组件，大小为色度的一半
    band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2; //一个block高
    band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;//一个block宽
    band->dct_b = DCTSIZE2;//  可以调节输出通道数 每个block大小
    long nb_elements =
        (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);//有多上个元素
    band->dct = new short[nb_elements];// 分配内存空间并初始化为 0
    std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
    return;
  }

  // prepare memory space dimensions // 准备内存空间的维度
  band->dct_h = srcinfo->comp_info[compNum].height_in_blocks;
  band->dct_w = srcinfo->comp_info[compNum].width_in_blocks;
  band->dct_b = DCTSIZE2;
  long nb_elements =
      (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);
  band->dct = new short[nb_elements];// 分配内存空间

  int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
  short unscale = 1;

  short *current_dct_coeff = band->dct;
  printf("band->dct_h:%d\n",band->dct_h);
  printf("band->dct_w:%d\n",band->dct_w);
  printf("band->dct_b:%d\n",band->dct_b);
  if(compNum==0){
      for(int i =0; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
                if (normalized) {// 如果需要标准化，则使用量化表进行反量化
                  unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
                }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
              //std::cout<<"-----------"<<std::endl;
        }
      i++;
      short* src2 = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
              if (normalized) {// 如果需要标准化，则使用量化表进行反量化
                unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
              }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
        }
      i=i+5;
    }
    for(int i =2; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
              if (normalized) {// 如果需要标准化，则使用量化表进行反量化
                unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
              }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
        }
      i++;
      short* src2 = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
              if (normalized) {// 如果需要标准化，则使用量化表进行反量化
                unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
              }
              *current_dct_coeff=src[j]*unscale;
              current_dct_coeff++;
        }
      i=i+5;
    }
  }
  else{
      for(int i =compNum+3; i<totalBlock;){
      short* src = inputBuffer + 64 * i;
      for (unsigned int j =0 ;j<64;j++){
              if (normalized) {// 如果需要标准化，则使用量化表进行反量化
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
  if (!warning_emitted) {// 发出警告，提示遇到非标准的 JPEG 图像，需要转码
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

  // transcode// 转码  读出来立马写回去
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

// 这个函数处理从start到end的MCU
void* process_mcus(void* args) {

    struct thread_data* data = (struct thread_data*)args;
    // 获取当前线程的真实 ID
    pid_t tid = syscall(SYS_gettid);

    // 设置 CPU 亲和性
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
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
  struct thread_data t1_data = {0, 17000 , srcinfo, mcu_buffer, block_buffer,7,1};
  struct thread_data t2_data = {17000, 24496, srcinfo, mcu_buffer, block_buffer,6,2};
  struct thread_data t3_data = {24496, 31900, srcinfo, mcu_buffer, block_buffer,5,3};
  struct thread_data t4_data = {31900, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,4,4};
  srcinfo->unread_marker = 0;
  srcinfo->now_b1_id=1;
  srcinfo->now_b2_id=srcinfo->huff_scan.buffer_id[17000]-1;//di er ge xian cheng kai shi de buffer id;
  srcinfo->now_b3_id=srcinfo->huff_scan.buffer_id[24496]-1;
  srcinfo->now_b4_id=srcinfo->huff_scan.buffer_id[31900]-1;
  (*srcinfo->src->retrun_file_start) (srcinfo);
  // 创建线程1
  pthread_create(&thread1, NULL, process_mcus, &t1_data);
  // 创建线程2
  pthread_create(&thread2, NULL, process_mcus, &t2_data);
  // 创建线程1
  pthread_create(&thread3, NULL, process_mcus, &t3_data);
  // 创建线程2
  pthread_create(&thread4, NULL, process_mcus, &t4_data);

  // 等待两个线程完成
  pthread_join(thread1, NULL);

  pthread_join(thread2, NULL);

  pthread_join(thread3, NULL);

  pthread_join(thread4, NULL);
  read_dct_coefficients(srcinfo, inputBuffer, 0, band1, normalized,totalBlock);// 读取第一个通道的 DCT 系数
  if (channels == 3) {// 如果通道数为 3，则继续读取第二个和第三个通道的 DCT 系数，否则将其设置为虚拟的 DCT 系数
    read_dct_coefficients(srcinfo, inputBuffer, 1, band2, normalized,totalBlock);
    read_dct_coefficients(srcinfo, inputBuffer, 2, band3, normalized,totalBlock);
  } else {
    dummy_dct_coefficients(band2);
    dummy_dct_coefficients(band3);
  }
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

  if (!is_grayscale(srcinfo) && !is_h2_v2(srcinfo)) {// 如果图像不是灰度且不是 H2V2 格式，则进行转码
    transcode(srcinfo, &buffer);
  }
  // 读取 JPEG 文件的 DCT 系数
  jpeg_read_coefficients_scan(srcinfo);
}
void read_dct_coefficients_from_file_mult(char *filename, bool normalized,
                                      int channels, band_info *band1,
                                      band_info *band2, band_info *band3,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
  read_dct_coefficients_from_srcinfo_mult(srcinfo,normalized,channels,band1,band2,band3,buffer);
  jpeg_abort_decompress(srcinfo);
  jpeg_destroy_decompress(srcinfo);
  fclose(infile);
}
void read_dct_coefficients_from_file(
    char *filename, bool normalized, int channels, short **band1_dct,
    int *band1_dct_h, int *band1_dct_w, int *band1_dct_b, short **band2_dct,
    int *band2_dct_h, int *band2_dct_w, int *band2_dct_b, short **band3_dct,
    int *band3_dct_h, int *band3_dct_w, int *band3_dct_b) {
  band_info band1, band2, band3;
  FILE *infile;
  jpeg_decompress_struct srcinfo;
  unsigned char *buffer = nullptr;
  read_dct_coefficients_from_file_scan(filename,infile,&srcinfo,buffer);
  read_dct_coefficients_from_file_mult(filename,normalized,channels,&band1,&band2,&band3,infile,&srcinfo,buffer);
  unpack_band_info(&band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
  unpack_band_info(&band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
  unpack_band_info(&band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
}

} // namespace common
} // namespace jpeg2dct