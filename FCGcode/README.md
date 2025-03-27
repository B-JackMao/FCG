1. Download and compile TensorFlow following the official TensorFlow source code instructions.  
   [TensorFlow GitHub - Benchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)

2. Replace the existing convolution execution functions in TensorFlow Lite with the `ruy` files. This includes replacing `trmul.cc` and `thread_pool.cc`, which are the kernel-level scheduling functions for convolution.

3. Replace the existing `lite` files in TensorFlow with new files, which contain code for heterogeneous decoding using CPU and GPU, as well as scheduling code for multi-JPEG image recognition.

4. Compile the `jpeg-9e` files according to the Libjpeg compilation method, referring to the official Libjpeg website's instructions for compiling on Android platforms. The multi-core Huffman decoding part in this project mainly involves modifying `jdhuff.c` and `jdatasrc.c` in `jpeg-9e`.  
   [Libjpeg GitHub](https://github.com/winlibs/libjpeg)

5. Move the generated `libjpeg.so` to the mobile phone.

6. Move the `libjpeg` folder into the `third_party` directory in TensorFlow, and add the `libjpeg` compilation environment dependency in the `WORKSPACE` file.

7. Compile and generate the benchmark test files. Move the files to the mobile phone and run tests.

8. The `jpeg_deep` folder contains the model training code. Please refer to [jpeg_deep GitHub](https://github.com/D3lt4lph4/jpeg_deep) for usage instructions.