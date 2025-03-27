
// #include <iostream>
// #include <fstream>
// #include <vector>

// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"

// void load_file(const std::string& file_path, std::vector<float>& data) {
//     std::ifstream file(file_path, std::ios::binary);
//     if (file) {
//         file.seekg(0, std::ios::end);
//         std::streampos file_size = file.tellg();
//         file.seekg(0, std::ios::beg);
        
//         data.resize(file_size);
//         file.read(reinterpret_cast<char*>(data.data()), file_size);
//         file.close();
//     } else {
//         std::cerr << "Failed to open file: " << file_path << std::endl;
//     }
// }

// int main() {
//     // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     std::string image_file = "preprocessed_image_3.raw";
//     std::string model_file = "R-34-C-C224.tflite";

//     // 1. Load preprocessed image data
//     std::vector<float> image_data;
//     load_file(image_file, image_data);

//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model =
//         tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     builder(&interpreter);

//     if (!interpreter) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter->AllocateTensors();

//     // Assuming input tensor is at index 0
//     int input_tensor_idx = interpreter->inputs()[0];
//     TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
//     std::memcpy(input_tensor->data.f, image_data.data(), image_data.size());

//     // 4. Run inference
//     interpreter->Invoke();

//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
// // 假设 output_tensor 是一个指向模型输出张量的指针
// // 假设 output_data_uint8 是指向量化输出数据的指针
// const float* output_data_uint8 = output_tensor->data.f;
// int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组

// // CIFAR-10 标签
// const char* cifar10_labels[] = {
//     "airplane", "automobile", "bird", "cat", "deer",
//     "dog", "frog", "horse", "ship", "truck"
// };

// // 找到具有最高置信度的类别
// float max_confidence = 0;
// int max_index = -1;
// for (int i = 0; i < output_size; ++i) {
//     if (output_data_uint8[i] > max_confidence) {
//         max_confidence = output_data_uint8[i];
//         max_index = i;
//     }
// }

// // 输出结果
// std::cout << "推断结果: ";
// for (int i = 0; i < output_size; ++i) {
//     std::cout << cifar10_labels[i] << ": " << static_cast<int>(output_data_uint8[i]) << " ";
// }
// std::cout << std::endl;

// if (max_index != -1) {
//     std::cout << "预测类别: " << cifar10_labels[max_index] << " 置信度: " << static_cast<int>(max_confidence) << std::endl;
// } else {
//     std::cerr << "没有做出预测。" << std::endl;
// }

//      //Example: Print the output
//     //std::cout << "Predicted class: " << std::distance(output_tensor->data.data, std::max_element(output_tensor->data.data, output_tensor->data.data + output_tensor->bytes / sizeof(float))) << std::endl;
//     //std::cout<<output_tensor->data.data<<std::endl;
//     return 0;
// }

// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <jpeglib.h>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"

// // Function to decode JPEG image using libjpeg
// bool decodeJPEG(const std::string& filename, std::vector<float>& imageData, int& width, int& height, int& channels) {
//     FILE* infile = fopen(filename.c_str(), "rb");
//     if (!infile) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return false;
//     }

//     jpeg_decompress_struct cinfo;
//     jpeg_error_mgr jerr;
//     cinfo.err = jpeg_std_error(&jerr);

//     jpeg_create_decompress(&cinfo);
//     jpeg_stdio_src(&cinfo, infile);
//     jpeg_read_header(&cinfo, TRUE);
//     jpeg_start_decompress(&cinfo);

//     width = cinfo.output_width;
//     height = cinfo.output_height;
//     channels = cinfo.output_components;

//     imageData.resize(width * height * channels);
//     float* rowptr = imageData.data();
//     while (cinfo.output_scanline < cinfo.output_height) {
//         jpeg_read_scanlines(&cinfo, &rowptr, 1);
//         rowptr += width * channels;
//     }

//     jpeg_finish_decompress(&cinfo);
//     jpeg_destroy_decompress(&cinfo);
//     fclose(infile);

//     return true;
// }
// // Function to add batch dimension
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }

// // Updated resizeImage function for NHWC format
// std::vector<float> resizeImage(const std::vector<float>& inputImage, int inputWidth, int inputHeight, int channels, int targetWidth, int targetHeight) {
//     std::vector<float> outputImage(targetHeight * targetWidth * channels);

//     // Simple nearest-neighbor resizing (could be replaced with better resizing algorithm)
//     for (int y = 0; y < targetHeight; ++y) {
//         for (int x = 0; x < targetWidth; ++x) {
//             int srcX = x * inputWidth / targetWidth;
//             int srcY = y * inputHeight / targetHeight;
//             for (int c = 0; c < channels; ++c) {
//                 outputImage[(y * targetWidth + x) * channels + c] = inputImage[(srcY * inputWidth + srcX) * channels + c];
//             }
//         }
//     }

//     return outputImage;
// }

// // Function to run inference using TFLite
// bool runInference(const std::vector<float>& imageData, int width, int height, int channels, const std::string& modelPath) {
//     int targetHeight = 224;
//     int targetWidth=224;
//     std::vector<float> resizedImage = resizeImage(imageData, width, height, channels, targetWidth, targetHeight);
//     std::vector<float> batchedImage = addBatchDimension(resizedImage, targetHeight, targetWidth, channels);
//         // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model =
//         tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     builder(&interpreter);

//     if (!interpreter) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter->AllocateTensors();
//     interpreter->SetNumThreads(4);
//     // Assuming input tensor is at index 0
//     int input_tensor_idx = interpreter->inputs()[0];
//     TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
//     std::memcpy(input_tensor->data.f, batchedImage.data(), batchedImage.size());


//     const int num_iterations = 50;
//     long long total_duration = 0;
//     for (int i = 0; i < num_iterations; ++i) {
//     auto start_invoke = std::chrono::high_resolution_clock::now();
//     // 4. Run inference
//     interpreter->Invoke();
//     auto end_invoke = std::chrono::high_resolution_clock::now();
//     auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke);
//     total_duration += duration_invoke.count();
//     }
//     // 输出执行时间
//     //std::cout << "invoke: " << duration_invoke.count() << " 毫秒" << std::endl;
//     double average_duration = static_cast<double>(total_duration) / num_iterations;

//     // 输出平均执行时间
//     std::cout << "平均执行时间: " << average_duration << " 毫秒" << std::endl;

//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
// // 假设 output_tensor 是一个指向模型输出张量的指针
// // 假设 output_data_uint8 是指向量化输出数据的指针
// const float* output_data_uint8 = output_tensor->data.f;
// int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组

// // CIFAR-10 标签
// const char* cifar10_labels[] = {
//     "airplane", "automobile", "bird", "cat", "deer",
//     "dog", "frog", "horse", "ship", "truck"
// };

// // 找到具有最高置信度的类别
// float max_confidence = 0;
// int max_index = -1;
// for (int i = 0; i < output_size; ++i) {
//     if (output_data_uint8[i] > max_confidence) {
//         max_confidence = output_data_uint8[i];
//         max_index = i;
//     }
// }

// // 输出结果
// std::cout << "推断结果: ";
// for (int i = 0; i < output_size; ++i) {
//     std::cout << cifar10_labels[i] << ": " << static_cast<int>(output_data_uint8[i]) << " ";
// }
// std::cout << std::endl;

// if (max_index != -1) {
//     std::cout << "预测类别: " << cifar10_labels[max_index] << " 置信度: " << static_cast<int>(max_confidence) << std::endl;
// } else {
//     std::cerr << "没有做出预测。" << std::endl;
// }


//     return true;
// }

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//     auto start_all = std::chrono::high_resolution_clock::now();
//     char* filename = argv[1];
//     std::string modelPath = "ResNet50.tflite";

//     std::vector<float> imageData;
//     int width, height, channels;
//     auto start_prepare_data = std::chrono::high_resolution_clock::now();
//     if (!decodeJPEG(filename, imageData, width, height, channels)) {
//         std::cerr << "Failed to decode JPEG image." << std::endl;
//         return -1;
//     }
//     auto end_prepare_data = std::chrono::high_resolution_clock::now();
//     auto duration_prepare_data = std::chrono::duration_cast<std::chrono::milliseconds>(end_prepare_data - start_prepare_data);

//     // 输出执行时间
//     std::cout << "prepare: " << duration_prepare_data.count() << " 毫秒" << std::endl;
//     if (!runInference(imageData, width, height, channels, modelPath)) {
//         std::cerr << "Failed to run inference." << std::endl;
//         return -1;
//     }
//     auto end_all = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);

//     // 输出执行时间
//     std::cout << "执行时间: " << duration.count() << " 毫秒" << std::endl;

//     return 0;
// }


// //san shu ru from txt
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <numeric>

// std::vector<float> readDataFromFile(const std::string& filename) {
//     std::ifstream file(filename);
//     std::vector<float> data;

//     if (file.is_open()) {
//         int value;
//         while (file >> value) {
//             data.push_back(static_cast<float>(value));
//         }
//         file.close();
//     } else {
//         std::cerr << "Failed to open file: " << filename << std::endl;
//     }

//     return data;
// }

// int main() {
//     // 加载TFLite模型
//     const char* model_path = "ResNet50_jpeg_late_concat.tflite";
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return 1;
//     }

//     // 创建解释器
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return 1;
//     }

//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return 1;
//     }

//     // 读取数据
//     std::vector<float> input_data1 = readDataFromFile("input1_data.txt");
//     std::vector<float> input_data2 = readDataFromFile("input2_data.txt");
//     std::vector<float> input_data3 = readDataFromFile("input3_data.txt");

//     // 获取输入张量的索引并设置值
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     int input_index3 = interpreter->inputs()[2];
//     // 确保输入张量的大小正确
//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     TfLiteTensor* tensor3 = interpreter->tensor(input_index3);


//     if (tensor1->bytes != input_data1.size() ||
//         tensor2->bytes != input_data2.size() ||
//         tensor3->bytes != input_data3.size() ) {
//         std::cerr << "Input tensor size does not match data size." << std::endl;
//         return 1;
//     }

//     std::memcpy(tensor1->data.f, input_data1.data(), input_data1.size());
//     std::memcpy(tensor2->data.f, input_data2.data(), input_data2.size());
//     std::memcpy(tensor3->data.f, input_data3.data(), input_data3.size());
// // 运行推理
//     if (interpreter->Invoke() != kTfLiteOk) {
//         std::cerr << "Failed to invoke interpreter." << std::endl;
//         return 1;
//     }
// // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     // 假设 output_tensor 是一个指向模型输出张量的指针
//     // 假设 output_data_uint8 是指向量化输出数据的指针
//     const float* output_data = output_tensor->data.f;
//    //float* output_data_uint8 = interpreter->typed_output_tensor<float>(output_tensor_idx);
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组

//     // 打印输出数据
//     std::cout << "Output data:" << std::endl;
//     for (size_t i = 0; i < interpreter->tensor(output_tensor_idx)->bytes; ++i) {
//         std::cout << static_cast<int>(output_data[i]) << " ";
//     }
//     std::cout << std::endl;

//     // CIFAR-10 标签
//     const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

//     // 找到概率得分最高的类别索引
//     int predicted_class_index = std::distance(output_data, std::max_element(output_data, output_data + interpreter->tensor(output_tensor_idx)->bytes));
//     const char* predicted_class_label = cifar10_labels[predicted_class_index];
//     int confidence_score = output_data[predicted_class_index];

//     // 打印结果
//     std::cout << "预测的类别：" << predicted_class_label << std::endl;
//     std::cout << "概率得分：" << confidence_score << std::endl;

//     return 0;
// }


// #include <chrono>
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <cstring>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// //#include "tensorflow/lite/tools/gen_op_registration.h"
// #include "dctfromjpg.h"

// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// //using namespace jpeg2dct::common;
// std::vector<float> convertToUint8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<float> uint8_vector;
//     long nb_elements = (long)(dct_h) * (long)(dct_w) * (long)(dct_b);
//     uint8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         uint8_vector.push_back(static_cast<float>(band_dct[i]));
//     }
//     return uint8_vector;
// }

// void preprocess_dct_coefficients(short *band_dct, int band_dct_h, int band_dct_w, int band_dct_b,
//                                  float *preprocessed_data, int size) {
//     // 剪辑到0-255范围内，并转换为uint8类型
//     for (int i = 0; i < size; ++i) {
//         int clipped_value = std::max(std::min(static_cast<int>(band_dct[i]), 255), 0);
//         preprocessed_data[i] = static_cast<float>(clipped_value);
//     }
// }

// void printTensorShape(TfLiteTensor* tensor) {
//     std::cout << "Tensor shape: [";
//     for (int i = 0; i < tensor->dims->size; ++i) {
//         std::cout << tensor->dims->data[i];
//         if (i < tensor->dims->size - 1) {
//             std::cout << ", ";
//         }
//     }
//     std::cout << "]" << std::endl;
// }

// // Function to write vector<float> data to a text file
// void writeVectorToFile(const std::vector<float>& data, const std::string& filename) {
//     std::ofstream file(filename);
//     if (file.is_open()) {
//         for (float value : data) {
//             file << static_cast<int>(value) << "\n"; // Writing each byte as integer value in a new line
//         }
//         file.close();
//         std::cout << "Data written to " << filename << std::endl;
//     } else {
//         std::cerr << "Unable to open file: " << filename << std::endl;
//     }
// }
// // 双线性插值函数
// float bilinearInterpolate(const std::vector<float>& input, int inputWidth, int inputHeight, float x, float y, int channel, int inputChannels) {
//     int x1 = static_cast<int>(x);
//     int y1 = static_cast<int>(y);
//     int x2 = std::min(x1 + 1, inputWidth - 1);
//     int y2 = std::min(y1 + 1, inputHeight - 1);

//     float x2_x = x2 - x;
//     float y2_y = y2 - y;
//     float x_x1 = x - x1;
//     float y_y1 = y - y1;

//     int idx11 = (y1 * inputWidth + x1) * inputChannels + channel;
//     int idx12 = (y1 * inputWidth + x2) * inputChannels + channel;
//     int idx21 = (y2 * inputWidth + x1) * inputChannels + channel;
//     int idx22 = (y2 * inputWidth + x2) * inputChannels + channel;

//     float result = 
//         input[idx11] * x2_x * y2_y +
//         input[idx12] * x_x1 * y2_y +
//         input[idx21] * x2_x * y_y1 +
//         input[idx22] * x_x1 * y_y1;

//     return static_cast<float>(result);
// }

// // 调整图像大小函数
// std::vector<float> resizeImage(
//     const std::vector<float>& inputData,
//     int inputWidth, int inputHeight, int inputChannels,
//     int outputWidth, int outputHeight
// ) {
//     std::vector<float> outputData(outputWidth * outputHeight * inputChannels);

//     for (int i = 0; i < outputHeight; ++i) {
//         for (int j = 0; j < outputWidth; ++j) {
//             float x_ratio = static_cast<float>(inputWidth) / outputWidth;
//             float y_ratio = static_cast<float>(inputHeight) / outputHeight;

//             float x = j * x_ratio;
//             float y = i * y_ratio;

//             for (int c = 0; c < inputChannels; ++c) {
//                 outputData[(i * outputWidth + j) * inputChannels + c] = bilinearInterpolate(inputData, inputWidth, inputHeight, x, y, c, inputChannels);
//             }
//         }
//     }

//     return outputData;
// }
// void runModelWithDCTData(short* band1_dct, int band1_dct_h, int band1_dct_w, int band1_dct_b,
//                          short* band2_dct, int band2_dct_h, int band2_dct_w, int band2_dct_b,
//                          short* band3_dct, int band3_dct_h, int band3_dct_w, int band3_dct_b) {
//     // 转换 float* 数据到 std::vector<float>
//     //std::vector<float> input1 = convertToUint8Vector(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
//     //std::vector<float> input2 = convertToUint8Vector(band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
//     //std::vector<float> input3 = convertToUint8Vector(band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
//     std::vector<float> input_data1(band1_dct_h * band1_dct_w * band1_dct_b);
//     std::vector<float> input_data2(band2_dct_h * band2_dct_w * band2_dct_b);
//     std::vector<float> input_data3(band3_dct_h * band3_dct_w * band3_dct_b);
//     preprocess_dct_coefficients(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b, input_data1.data(), input_data1.size());
//     preprocess_dct_coefficients(band2_dct, band2_dct_h, band2_dct_w, band2_dct_b, input_data2.data(), input_data2.size());
//     preprocess_dct_coefficients(band3_dct, band3_dct_h, band3_dct_w, band3_dct_b, input_data3.data(), input_data3.size());

//     int targetWidth1 = 28;
//     int targetHeight1= 28;
//     int targetWidth2 = 14;
//     int targetHeight2 = 14;
//     std::vector<float> resizedImage1 = resizeImage(input_data1, band1_dct_w, band1_dct_h, band1_dct_b, targetWidth1, targetHeight1);
//     std::vector<float> resizedImage2 = resizeImage(input_data2, band2_dct_w, band2_dct_h, band2_dct_b, targetWidth2, targetHeight2);
//     std::vector<float> resizedImage3 = resizeImage(input_data3, band3_dct_w, band3_dct_h, band3_dct_b, targetWidth2, targetHeight2);


//     std::vector<float> input1_batch = addBatchDimension(resizedImage1,targetHeight1, targetWidth1, band1_dct_b);
//     std::vector<float> input2_batch = addBatchDimension(resizedImage2,targetHeight2, targetWidth2, band2_dct_b);
//     std::vector<float> input3_batch = addBatchDimension(resizedImage3,targetHeight2, targetWidth2, band3_dct_b);
//     // Write each vector to a separate file
//     // writeVectorToFile(input1_batch, "input1_data.txt");
//     // writeVectorToFile(input2_batch, "input2_data.txt");
//     // writeVectorToFile(input3_batch, "input3_data.txt");
//     // 加载模型
//     const char* model_path = "ResNet50_jpeg_late_concat_long.tflite";
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return;
//     }
//     // 创建解释器
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//         return;
//     }
//     interpreter->SetNumThreads(4);
//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return;
//     }
    
//     // 获取输入张量索引
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     int input_index3 = interpreter->inputs()[2];

//     // 确保输入张量的大小正确
//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     TfLiteTensor* tensor3 = interpreter->tensor(input_index3);


//     if (tensor1->bytes != input1_batch.size() ||
//         tensor2->bytes != input2_batch.size() ||
//         tensor3->bytes != input3_batch.size() ) {
//         std::cerr << "Input tensor size does not match data size." << std::endl;
//         return;
//     }

//     std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.f, input2_batch.data(), input2_batch.size());
//     std::memcpy(tensor3->data.f, input3_batch.data(), input3_batch.size());
    
//     const int num_iterations = 50;
//     long long total_duration = 0;
//     for (int i = 0; i < num_iterations; ++i) {
//     auto start_invoke = std::chrono::high_resolution_clock::now();
//     // 4. Run inference
//     interpreter->Invoke();
//     auto end_invoke = std::chrono::high_resolution_clock::now();
//     auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke);
//     total_duration += duration_invoke.count();
//     }
//     // 输出执行时间
//     //std::cout << "invoke: " << duration_invoke.count() << " 毫秒" << std::endl;
//     double average_duration = static_cast<double>(total_duration) / num_iterations;

//     // 输出平均执行时间
//     std::cout << "平均执行时间: " << average_duration << " 毫秒" << std::endl;

//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     // 假设 output_data_uint8 是指向量化输出数据的指针
//     const float* output_data= output_tensor->data.f;
//    //float* output_data_uint8 = interpreter->typed_output_tensor<float>(output_tensor_idx);
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组

//         // 打印输出数据
//     std::cout << "Output data:" << std::endl;
//     for (size_t i = 0; i < interpreter->tensor(output_tensor_idx)->bytes; ++i) {
//         std::cout << static_cast<int>(output_data[i]) << " ";
//     }
//     std::cout << std::endl;

//     // CIFAR-10 标签
//     const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

//     // 找到概率得分最高的类别索引
//     int predicted_class_index = std::distance(output_data, std::max_element(output_data, output_data + interpreter->tensor(output_tensor_idx)->bytes));
//     const char* predicted_class_label = cifar10_labels[predicted_class_index];
//     int confidence_score = output_data[predicted_class_index];

//     // 打印结果
//     std::cout << "预测的类别：" << predicted_class_label << std::endl;
//     std::cout << "概率得分：" << confidence_score << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }

//     char* filename = argv[1];
//     auto start_all = std::chrono::high_resolution_clock::now();
//     short *band1_dct, *band2_dct, *band3_dct;
//     int band1_dct_h, band1_dct_w, band1_dct_b;
//     int band2_dct_h, band2_dct_w, band2_dct_b;
//     int band3_dct_h, band3_dct_w, band3_dct_b;
//     auto start_prepare_data = std::chrono::high_resolution_clock::now();
//     // 假设 read_dct_coefficients_from_file 已经实现并且返回正确的数据
//     jpeg2dct::common::read_dct_coefficients_from_file(filename, true, 3,
//                                                       &band1_dct, &band1_dct_h, &band1_dct_w, &band1_dct_b,
//                                                       &band2_dct, &band2_dct_h, &band2_dct_w, &band2_dct_b,
//                                                       &band3_dct, &band3_dct_h, &band3_dct_w, &band3_dct_b);
//     auto end_prepare_data = std::chrono::high_resolution_clock::now();
//     auto duration_prepare_data = std::chrono::duration_cast<std::chrono::milliseconds>(end_prepare_data - start_prepare_data);
//     std::cout << "prepare: " << duration_prepare_data.count() << " 毫秒" << std::endl;
//     // 运行模型
//     runModelWithDCTData(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b,
//                         band2_dct, band2_dct_h, band2_dct_w, band2_dct_b,
//                         band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
//     auto end_all = std::chrono::high_resolution_clock::now();
//     // 计算持续时间并转换为毫秒
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);

//     // 清理内存
//     delete[] band1_dct;
//     delete[] band2_dct;
//     delete[] band3_dct;
//         // 输出执行时间
//     std::cout << "all: " << duration.count() << " 毫秒" << std::endl;

//     return 0;
// }

// #include <algorithm>
// #include <thread>
// #include <chrono>
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <cstring>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"
// //#include "tensorflow/lite/tools/gen_op_registration.h"
// #include "dctfromjpg.h"
// #include "ctpl_stl.h"

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// //using namespace jpeg2dct::common;
// std::vector<float> convertToUint8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<float> uint8_vector;
//     long nb_elements = (long)(dct_h) * (long)(dct_w) * (long)(dct_b);
//     uint8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         uint8_vector.push_back(static_cast<float>(band_dct[i]));
//     }
//     return uint8_vector;
// }
// void printTensorShape(TfLiteTensor* tensor) {
//     std::cout << "Tensor shape: [";
//     for (int i = 0; i < tensor->dims->size; ++i) {
//         std::cout << tensor->dims->data[i];
//         if (i < tensor->dims->size - 1) {
//             std::cout << ", ";
//         }
//     }
//     std::cout << "]" << std::endl;
// }
// // Updated resizeImage function for NHWC format
// std::vector<float> resizeImage(const std::vector<float>& inputImage, int inputWidth, int inputHeight, int channels, int targetWidth, int targetHeight) {
//     std::vector<float> outputImage(targetHeight * targetWidth * channels);
//     for (int y = 0; y < targetHeight; ++y) {
//         float fy = static_cast<float>(y) * (inputHeight - 1) / (targetHeight - 1);
//         int y0 = static_cast<int>(fy);
//         int y1 = std::min(y0 + 1, inputHeight - 1);
//         float ly = fy - y0;
//         float hy = 1.0f - ly;
//         for (int x = 0; x < targetWidth; ++x) {
//             float fx = static_cast<float>(x) * (inputWidth - 1) / (targetWidth - 1);
//             int x0 = static_cast<int>(fx);
//             int x1 = std::min(x0 + 1, inputWidth - 1);
//             float lx = fx - x0;
//             float hx = 1.0f - lx;
//             for (int c = 0; c < channels; ++c) {
//                 float value = (inputImage[(y0 * inputWidth + x0) * channels + c] * hx + 
//                                inputImage[(y0 * inputWidth + x1) * channels + c] * lx) * hy +
//                               (inputImage[(y1 * inputWidth + x0) * channels + c] * hx + 
//                                inputImage[(y1 * inputWidth + x1) * channels + c] * lx) * ly;
//                 outputImage[(y * targetWidth + x) * channels + c] = static_cast<float>(value);
//             }
//         }
//     }
//     return outputImage;
// }

// const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};


// void runInference(const std::vector<float>& input1_batch, 
//                               const std::vector<float>& input2_batch, 
//                               const std::vector<float>& input3_batch) {
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     int input_index3 = interpreter->inputs()[2];

//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     TfLiteTensor* tensor3 = interpreter->tensor(input_index3);
//     std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.f, input2_batch.data(), input2_batch.size());
//     std::memcpy(tensor3->data.f, input3_batch.data(), input3_batch.size());
//     // 4. Run inference
//     interpreter->Invoke();

//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     // 假设 output_data_uint8 是指向量化输出数据的指针
//     const float* output_data= output_tensor->data.f;
// //float* output_data_uint8 = interpreter->typed_output_tensor<float>(output_tensor_idx);
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组

//         // 打印输出数据
//     // std::cout << "Output data:" << std::endl;
//     // for (size_t i = 0; i < interpreter->tensor(output_tensor_idx)->bytes; ++i) {
//     //     std::cout << static_cast<int>(output_data[i]) << " ";
//     // }
//     // std::cout << std::endl;

//     // // CIFAR-10 标签
//     // const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

//     // // 找到概率得分最高的类别索引
//     // int predicted_class_index = std::distance(output_data, std::max_element(output_data, output_data + interpreter->tensor(output_tensor_idx)->bytes));
//     // const char* predicted_class_label = cifar10_labels[predicted_class_index];
//     // int confidence_score = output_data[predicted_class_index];
//     // // 打印结果
//     // std::cout << "预测的类别：" << predicted_class_label << std::endl;
//     // std::cout << "概率得分：" << confidence_score << std::endl;

// }
// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//     const char* model_path = "ResNet50_jpeg_late_concat_long_cg.tflite";
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return 0;
//     }
//     // 创建解释器
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     //std::unique_ptr<tflite::Interpreter> interpreter;
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//         return 0;
//     }
//     // 创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }

//     interpreter->SetNumThreads(3);
//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return 0;
//     }
//     ctpl::thread_pool cpu1_thread_(1,6);
//     std::future<void> cpu1_future;

// char* files[8] = {
//     "1_jiqixuexi_1000.jpeg","1_jiqixuexi_2000.jpeg","1_jiqixuexi_3000.jpeg","1_jiqixuexi_4000.jpeg","2_jiqixuexi_1000.jpeg","2_jiqixuexi_2000.jpeg","2_jiqixuexi_3000.jpeg","2_jiqixuexi_4000.jpeg"};


// for (int  j =0;j<4;j++){
//     auto start_all = std::chrono::high_resolution_clock::now();
//     const int num_iterations = 56;
//     long long total_duration = 0;
//     char* filename = files[j];
//     for (int i = 0; i < num_iterations; ++i) {
        
        
//         auto start_prepare_data = std::chrono::high_resolution_clock::now();
//         short *band1_dct, *band2_dct, *band3_dct;
//         int band1_dct_h, band1_dct_w, band1_dct_b;
//         int band2_dct_h, band2_dct_w, band2_dct_b;
//         int band3_dct_h, band3_dct_w, band3_dct_b;
        
//         // 假设 read_dct_coefficients_from_file 已经实现并且返回正确的数据
//         jpeg2dct::common::read_dct_coefficients_from_file(filename, true, 3,
//                                                         &band1_dct, &band1_dct_h, &band1_dct_w, &band1_dct_b,
//                                                         &band2_dct, &band2_dct_h, &band2_dct_w, &band2_dct_b,
//                                                         &band3_dct, &band3_dct_h, &band3_dct_w, &band3_dct_b);
        
//         std::vector<float> input1 = convertToUint8Vector(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
//         std::vector<float> input2 = convertToUint8Vector(band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
//         std::vector<float> input3 = convertToUint8Vector(band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
//         int targetWidth1 = 28;
//         int targetHeight1= 28;
//         int targetWidth2 = 14;
//         int targetHeight2 = 14;
//         std::vector<float> resizedImage1 = resizeImage(input1, band1_dct_w, band1_dct_h, band1_dct_b, targetWidth1, targetHeight1);
//         std::vector<float> resizedImage2 = resizeImage(input2, band2_dct_w, band2_dct_h, band2_dct_b, targetWidth2, targetHeight2);
//         std::vector<float> resizedImage3 = resizeImage(input3, band3_dct_w, band3_dct_h, band3_dct_b, targetWidth2, targetHeight2);

//         std::vector<float> input1_batch = addBatchDimension(resizedImage1,targetHeight1, targetWidth1, band1_dct_b);
//         std::vector<float> input2_batch = addBatchDimension(resizedImage2,targetHeight2, targetWidth2, band2_dct_b);
//         std::vector<float> input3_batch = addBatchDimension(resizedImage3,targetHeight2, targetWidth2, band3_dct_b);
//         auto end_prepare_data = std::chrono::high_resolution_clock::now();
//         if (cpu1_future.valid()) {
//             cpu1_future.wait();  
//         }

//         cpu1_future = cpu1_thread_.push([&input1_batch,&input2_batch, &input3_batch](int) { 
//             runInference(input1_batch, input2_batch, input3_batch); // CPU1
//         });
//         if(i==num_iterations-1){
//         cpu1_future.wait();
//         }
//         // cpu1_future.wait();
        
//         //runInference(input1_batch, input2_batch, input3_batch);

        
//         // 计算持续时间并转换为毫秒
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_prepare_data - start_prepare_data);
//         total_duration += duration.count();
//         // 清理内存
//         // delete[] band1_dct;
//         // delete[] band2_dct;
//         // delete[] band3_dct;
//     }
//     double average_duration = static_cast<double>(total_duration) / num_iterations;
//     auto end_all = std::chrono::high_resolution_clock::now();
//     auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);

//     // 输出平均执行时间
//     std::cout <<"num:"<<j<<" ,invoke:"<<duration_all.count()/num_iterations<< ",prapare: " << average_duration << " 毫秒" << std::endl;
//     }
//     return 0;
// }
/*-------------------------------DCT-CG-liu shui-danhe huffman-------------------------------*/
// #include <algorithm>
// #include <thread>
// #include <chrono>
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <cstring>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"
// //#include "tensorflow/lite/tools/gen_op_registration.h"
// #include "dctfromjpg.h"
// #include "ctpl_stl.h"

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// //using namespace jpeg2dct::common;
// std::vector<float> convertToUint8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<float> uint8_vector;
//     long nb_elements = (long)(dct_h) * (long)(dct_w) * (long)(dct_b);
//     uint8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         uint8_vector.push_back(static_cast<float>(band_dct[i]));
//     }
//     return uint8_vector;
// }

// // Updated resizeImage function for NHWC format
// short* resizeShortData(const short* input_data, int original_height, int original_width, int channels,
//                        int new_height, int new_width) {
//     // 动态分配内存给 resized_data
//     short* resized_data = new short[new_height * new_width * channels];

//     // 计算缩放比例
//     float scale_h = static_cast<float>(original_height - 1) / (new_height - 1);
//     float scale_w = static_cast<float>(original_width - 1) / (new_width - 1);

//     // 对每个像素进行插值
//     for (int y = 0; y < new_height; ++y) {
//         for (int x = 0; x < new_width; ++x) {
//             float gy = y * scale_h;
//             float gx = x * scale_w;

//             int gxi = static_cast<int>(gx);
//             int gyi = static_cast<int>(gy);

//             for (int c = 0; c < channels; ++c) {
//                 float c00 = input_data[(gyi * original_width + gxi) * channels + c];
//                 float c10 = input_data[(gyi * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];
//                 float c01 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + gxi) * channels + c];
//                 float c11 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];

//                 float cx0 = c00 + (c10 - c00) * (gx - gxi);
//                 float cx1 = c01 + (c11 - c01) * (gx - gxi);
//                 float cxy = cx0 + (cx1 - cx0) * (gy - gyi);

//                 resized_data[(y * new_width + x) * channels + c] = static_cast<short>(std::round(cxy));
//             }
//         }
//     }

//     return resized_data;
// }

// const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

// const int height = 14;
// const int width = 14;
// const int depth1 = 64;  // 第一个输入的通道数
// const int depth2 = 64;  // 第二个输入的通道数
// // 合并函数，将两个 14x14x64 的输入合并为一个 14x14x128
// std::vector<float> mergeInputs(const std::vector<float>& input1, const std::vector<float>& input2) {
//     // 检查输入大小是否匹配
//     if (input1.size() != height * width * depth1 || input2.size() != height * width * depth2) {
//         std::cerr << "输入大小不匹配!" << std::endl;
//         return {};
//     }

//     // 创建合并后的 vector，大小为14x14x128
//     std::vector<float> merged_input(height * width * (depth1 + depth2));

//     // 逐元素合并
//     for (int i = 0; i < height * width; ++i) {
//         // 将 input1 的数据拷贝到 merged_input 中
//         for (int d1 = 0; d1 < depth1; ++d1) {
//             merged_input[i * (depth1 + depth2) + d1] = input1[i * depth1 + d1];
//         }
//         // 将 input2 的数据拷贝到 merged_input 中
//         for (int d2 = 0; d2 < depth2; ++d2) {
//             merged_input[i * (depth1 + depth2) + depth1 + d2] = input2[i * depth2 + d2];
//         }
//     }

//     return merged_input;
// }

// void runInference(const std::vector<float>& input1_batch, 
//                               const std::vector<float>& input2_batch, 
//                               const std::vector<float>& input3_batch) {
//     std::vector<float> merged_input = mergeInputs(input2_batch, input3_batch);
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     //int input_index3 = interpreter->inputs()[2];

//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     //TfLiteTensor* tensor3 = interpreter->tensor(input_index3);
//     //auto start_all_invoke = std::chrono::high_resolution_clock::now();
//     std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.f, merged_input.data(), merged_input.size());
//     //std::memcpy(tensor3->data.f, input3_batch.data(), input3_batch.size());
//     // 4. Run inference
//     interpreter->Invoke();
    
//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     // 假设 output_data_uint8 是指向量化输出数据的指针
//     const float* output_data= output_tensor->data.f;
// //float* output_data_uint8 = interpreter->typed_output_tensor<float>(output_tensor_idx);
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组
// }
// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//     const char* model_path = argv[1];
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return 0;
//     }
//     // 创建解释器
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//         return 0;
//     }
// //创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }
//     interpreter->SetNumThreads(3);
//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return 0;
//     }
//     ctpl::thread_pool cpu1_thread_(1,7);
//     std::future<void> cpu1_future;

//     char* files[8] = {
//         "83.jpeg", "84.jpeg", "85.jpeg",
//         "86.jpeg", "87.jpeg", "88.jpeg",
//         "89.jpeg", "90.jpeg"
//     };
//     auto start_all = std::chrono::high_resolution_clock::now();
//     const int num_iterations = 50;
//       long long total_duration = 0;
//       long long total_duration_invoke = 0;
//       long long total_duration_all = 0;
//     for (int i = 0; i < num_iterations; ++i) {

//         auto start_prepare_data = std::chrono::high_resolution_clock::now();
//         char* filename = files[i%8];
//         short *band1_dct, *band2_dct, *band3_dct;
//         int band1_dct_h, band1_dct_w, band1_dct_b;
//         int band2_dct_h, band2_dct_w, band2_dct_b;
//         int band3_dct_h, band3_dct_w, band3_dct_b;
        
//         // 假设 read_dct_coefficients_from_file 已经实现并且返回正确的数据
//         jpeg2dct::common::read_dct_coefficients_from_file(filename, true, 3,
//                                                         &band1_dct, &band1_dct_h, &band1_dct_w, &band1_dct_b,
//                                                         &band2_dct, &band2_dct_h, &band2_dct_w, &band2_dct_b,
//                                                         &band3_dct, &band3_dct_h, &band3_dct_w, &band3_dct_b);
        
//         int targetWidth1 = 28;
//         int targetHeight1= 28;
//         int targetWidth2 = 14;
//         int targetHeight2 = 14;
//         short* resized_data1 = resizeShortData(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b, targetWidth1, targetHeight1);
//         short* resized_data2 = resizeShortData(band2_dct, band2_dct_h, band2_dct_w, band2_dct_b, targetWidth2, targetHeight2);
//         short* resized_data3 = resizeShortData(band3_dct, band3_dct_h, band3_dct_w, band3_dct_b, targetWidth2, targetHeight2);

//         std::vector<float> input1 = convertToUint8Vector(band1_dct, targetWidth1, targetHeight1, band1_dct_b);
//         std::vector<float> input2 = convertToUint8Vector(band2_dct, targetWidth2, targetHeight2, band2_dct_b);
//         std::vector<float> input3 = convertToUint8Vector(band3_dct, targetWidth2, targetHeight2, band3_dct_b);

//         std::vector<float> input1_batch = addBatchDimension(input1,targetHeight1, targetWidth1, band1_dct_b);
//         std::vector<float> input2_batch = addBatchDimension(input2,targetHeight2, targetWidth2, band2_dct_b);
//         std::vector<float> input3_batch = addBatchDimension(input3,targetHeight2, targetWidth2, band3_dct_b);
//         auto end_all_prepare = std::chrono::high_resolution_clock::now();
//         if (cpu1_future.valid()) {
//             cpu1_future.wait();  
//         }

//         cpu1_future = cpu1_thread_.push([&input1_batch,&input2_batch, &input3_batch](int) { 
//             runInference(input1_batch, input2_batch, input3_batch); // CPU1
//         });
//         if(i==num_iterations-1){
//         cpu1_future.wait();
//         }
//         //runInference(input1_batch, input2_batch, input3_batch);
//         auto end_all = std::chrono::high_resolution_clock::now();
//         delete[] band1_dct;
//         delete[] band2_dct;
//         delete[] band3_dct;
//         // 计算持续时间并转换为毫秒
//         // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_all_prepare - start_prepare_data);
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all_prepare - start_prepare_data);
//         auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - end_all_prepare);
//         auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_prepare_data);
//          total_duration += duration.count();
//          total_duration_invoke +=duration_invoke.count();
//          total_duration_all +=duration_all.count();
//     }
//     //outFile.close();
//      auto end_all = std::chrono::high_resolution_clock::now();
//      auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
//     std::cout << "average 时间: " << total_duration_all /num_iterations<< " 毫秒" << std::endl;
//     std::cout << "all时间: " << total_duration_all << " 毫秒" << std::endl;
//     std::cout << " per:"<<total_duration/num_iterations<<std::endl;
//     std::cout<<"invoke:"<<total_duration_invoke/num_iterations<<std::endl;
    
//     return 0;
// }

/*--------------------------RGB-CG-chuan xing -liu shui----------------------------------*/
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
#include <atomic>
#include <functional>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "ctpl_stl.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
std::unique_ptr<tflite::Interpreter> interpreter;
// Function to decode JPEG image using libjpeg
bool decodeJPEG(const std::string& filename, std::vector<float>& imageData, int& width, int& height, int& channels) {
    cpu_set_t mask;
    CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
    CPU_SET(7, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    } 
    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    width = cinfo.output_width;
    height = cinfo.output_height;
    channels = cinfo.output_components;

    imageData.resize(width * height * channels);
    float* rowptr = imageData.data();
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
        rowptr += width * channels;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return true;
}
// Function to add batch dimension
std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
    std::vector<float> batchedImage(1 * height * width * channels);
    std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
    return batchedImage;
}

// Updated resizeImage function for NHWC format
std::vector<float> resizeImage(const std::vector<float>& inputImage, int inputWidth, int inputHeight, int channels, int targetWidth, int targetHeight) {
    std::vector<float> outputImage(targetHeight * targetWidth * channels);

    // Simple nearest-neighbor resizing (could be replaced with better resizing algorithm)
    for (int y = 0; y < targetHeight; ++y) {
        for (int x = 0; x < targetWidth; ++x) {
            int srcX = x * inputWidth / targetWidth;
            int srcY = y * inputHeight / targetHeight;
            for (int c = 0; c < channels; ++c) {
                outputImage[(y * targetWidth + x) * channels + c] = inputImage[(srcY * inputWidth + srcX) * channels + c];
            }
        }
    }

    return outputImage;
}
void inferenceTask(const std::vector<float>& imageData, int width, int height, int channels) {
    int targetHeight = 224;
    int targetWidth = 224;
    std::vector<float> resizedImage = resizeImage(imageData, width, height, channels, targetWidth, targetHeight);
    std::vector<float> batchedImage = addBatchDimension(resizedImage, targetHeight, targetWidth, channels);

    // Assuming input_tensor is properly set
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    std::memcpy(input_tensor->data.f, batchedImage.data(), batchedImage.size());

    auto start_invoke = std::chrono::high_resolution_clock::now();
    interpreter->Invoke();
    auto end_invoke = std::chrono::high_resolution_clock::now();

    int output_tensor_idx = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
    const float* output_data_uint8 = output_tensor->data.f;
    int output_size = output_tensor->dims->data[1];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
        // 替换为你的预处理后图像文件路径和TFLite模型文件路径
    // 2. Initialize TensorFlow Lite interpreter
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    ctpl::thread_pool cpu1_thread_(1,6);
    std::future<void> cpu1_future;
    builder(&interpreter);
    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    // 创建并绑定代理
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
    interpreter->SetNumThreads(4);
    // Assuming input tensor is at index 0
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);

    //char* filename = argv[1];
//     char* files[56] = {
//     "1_resized_image.jpeg", "2_resized_image.jpeg", "3_resized_image.jpeg", 
//     "4_resized_image.jpeg", "5_resized_image.jpeg", "6_resized_image.jpeg", 
//     "7_resized_image.jpeg", "8_resized_image.jpeg", "9_resized_image.jpeg", 
//     "10_resized_image.jpeg", "11_resized_image.jpeg", "12_resized_image.jpeg", 
//     "13_resized_image.jpeg", "14_resized_image.jpeg", "15_resized_image.jpeg", 
//     "16_resized_image.jpeg", "17_resized_image.jpeg", "18_resized_image.jpeg", 
//     "19_resized_image.jpeg", "20_resized_image.jpeg", "21_resized_image.jpeg", 
//     "22_resized_image.jpeg", "23_resized_image.jpeg", "24_resized_image.jpeg", 
//     "25_resized_image.jpeg", "26_resized_image.jpeg", "27_resized_image.jpeg", 
//     "28_resized_image.jpeg", "29_resized_image.jpeg", "30_resized_image.jpeg", 
//     "31_resized_image.jpeg", "32_resized_image.jpeg", "33_resized_image.jpeg", 
//     "34_resized_image.jpeg", "35_resized_image.jpeg", "36_resized_image.jpeg", 
//     "37_resized_image.jpeg", "38_resized_image.jpeg", "39_resized_image.jpeg", 
//     "40_resized_image.jpeg", "41_resized_image.jpeg", "42_resized_image.jpeg", 
//     "43_resized_image.jpeg", "44_resized_image.jpeg", "45_resized_image.jpeg", 
//     "46_resized_image.jpeg", "47_resized_image.jpeg", "48_resized_image.jpeg", 
//     "49_resized_image.jpeg", "50_resized_image.jpeg", "51_resized_image.jpeg", 
//     "52_resized_image.jpeg", "53_resized_image.jpeg", "54_resized_image.jpeg", 
//     "55_resized_image.jpeg", "56_resized_image.jpeg"
// };
    char* files[8] = {
        "83.jpeg", "84.jpeg", "85.jpeg",
        "86.jpeg", "87.jpeg", "88.jpeg",
        "89.jpeg", "90.jpeg"
    };
 std::vector<float> imageData;
    int width, height, channels;
    const int num_iterations = 50;
    long long total_duration = 0;
    long long total_invoke = 0;
    long long total_prepare = 0;
    for (int i = 0; i < num_iterations; ++i) {
        char* filename = files[i%8];
        auto start_prepare_data = std::chrono::high_resolution_clock::now();
        if (!decodeJPEG(filename, imageData, width, height, channels)) {
            std::cerr << "Failed to decode JPEG image." << std::endl;
            return -1;
        }

        // if (cpu1_future.valid()) {
        //     cpu1_future.wait();
        // }
        // cpu1_future = cpu1_thread_.push([&imageData, width, height, channels](int) { 
        //     inferenceTask(imageData, width, height, channels); // CPU1
        // });
        // if(i==num_iterations-1){
        // cpu1_future.wait();
        // }
        auto start_invoke = std::chrono::high_resolution_clock::now();
        inferenceTask(imageData, width, height, channels);
        auto end_invoke = std::chrono::high_resolution_clock::now();
        auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_prepare_data);
        auto duration_prepare = std::chrono::duration_cast<std::chrono::milliseconds>(start_invoke - start_prepare_data);
        auto duration_i = std::chrono::duration_cast<std::chrono::milliseconds>(end_invoke - start_invoke);
        total_duration += duration_invoke.count();
        total_prepare += duration_prepare.count();
        total_invoke += duration_i.count();
    }
    // 输出执行时间
    //std::cout << "invoke: " << duration_invoke.count() << " 毫秒" << std::endl;
    double average_duration = static_cast<double>(total_duration) / num_iterations;
    std::cout<<"jpeg:"<<total_prepare/num_iterations<<std::endl;
    std::cout<<"invoke:"<<total_invoke/num_iterations<<std::endl;
    // 输出平均执行时间
    std::cout << "平均执行时间: " << average_duration << " 毫秒" << std::endl;
    // 输出执行时间
    std::cout << "执行时间: " << total_duration << " 毫秒" << std::endl;

    return 0;
}



/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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


/*------------------------------------------End Three RGB Pipeline------------------------------------*/
// #include <thread>
// #include <sched.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <jpeglib.h>
// #include <future>
// #include <mutex>
// #include <condition_variable>
// #include <unordered_map>
// #include <atomic>
// #include <queue>
// #include <functional>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "ctpl_stl.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::unique_ptr<tflite::Interpreter> interpreter2;
// std::mutex mtx;
// std::condition_variable cv;
// std::queue<std::tuple<std::string, std::vector<float>, int, int, int,int>> task_part_queue[3]; // Task queue for each part
// std::unordered_map<int, int> task_state; // Tracks task state (0: part 1, 1: part 2, 2: part 3)
// bool done = false;

// // Function to decode JPEG image using libjpeg
// bool decodeJPEG(const std::string& filename, std::vector<float>& imageData, int& width, int& height, int& channels) {
//     FILE* infile = fopen(filename.c_str(), "rb");
//     if (!infile) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return false;
//     }

//     jpeg_decompress_struct cinfo;
//     jpeg_error_mgr jerr;
//     cinfo.err = jpeg_std_error(&jerr);

//     jpeg_create_decompress(&cinfo);
//     jpeg_stdio_src(&cinfo, infile);
//     jpeg_read_header(&cinfo, TRUE);
//     jpeg_start_decompress(&cinfo);

//     width = cinfo.output_width;
//     height = cinfo.output_height;
//     channels = cinfo.output_components;

//     imageData.resize(width * height * channels);
//     float* rowptr = imageData.data();
//     while (cinfo.output_scanline < cinfo.output_height) {
//         jpeg_read_scanlines(&cinfo, &rowptr, 1);
//         rowptr += width * channels;
//     }
//     std::cout<<"decode:"<<width<<"img:"<<&imageData<<std::endl;
//     jpeg_finish_decompress(&cinfo);
//     jpeg_destroy_decompress(&cinfo);
//     fclose(infile);

//     return true;
// }

// // Function to add batch dimension
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }

// // Updated resizeImage function for NHWC format
// std::vector<float> resizeImage(const std::vector<float>& inputImage, int inputWidth, int inputHeight, int channels, int targetWidth, int targetHeight) {
//     std::vector<float> outputImage(targetHeight * targetWidth * channels);

//     // Simple nearest-neighbor resizing (could be replaced with better resizing algorithm)
//     for (int y = 0; y < targetHeight; ++y) {
//         for (int x = 0; x < targetWidth; ++x) {
//             int srcX = x * inputWidth / targetWidth;
//             int srcY = y * inputHeight / targetHeight;
//             for (int c = 0; c < channels; ++c) {
//                 outputImage[(y * targetWidth + x) * channels + c] = inputImage[(srcY * inputWidth + srcX) * channels + c];
//             }
//         }
//     }

//     return outputImage;
// }

// void inferenceTask(const std::vector<float>& imageData, int width, int height, int channels,int id) {
//     int targetHeight = 224;
//     int targetWidth = 224;
//     std::cout<<"inf"<<width<<"img:"<<&imageData<<std::endl;
//     std::vector<float> resizedImage = resizeImage(imageData, width, height, channels, targetWidth, targetHeight);
//     std::vector<float> batchedImage = addBatchDimension(resizedImage, targetHeight, targetWidth, channels);
//     if(id%2==0){
//     // Assuming input_tensor is properly set
//     TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
//     std::memcpy(input_tensor->data.f, batchedImage.data(), batchedImage.size());

//     interpreter->Invoke();
//     std::cout<<"env"<<std::endl;
//     }
//     else{
//       // Assuming input_tensor is properly set
//     TfLiteTensor* input_tensor = interpreter2->tensor(interpreter2->inputs()[0]);
//     std::memcpy(input_tensor->data.f, batchedImage.data(), batchedImage.size());

//     interpreter2->Invoke();
//     std::cout<<"odd"<<std::endl;
//     }
// }

// void inferenceTask2(int id) {
//   if(id%2==0){
//     interpreter->Invoke_2();
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     const float* output_data_uint8 = output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1];
//   }
//   else{
//     interpreter2->Invoke_2();
//     int output_tensor_idx = interpreter2->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter2->tensor(output_tensor_idx);
//     const float* output_data_uint8 = output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1];
//   }
// }

// // Worker function for part 1
// void worker_decode(int cpu_id) {
//     while (true) {
//         std::tuple<std::string, std::vector<float>, int, int, int,int> task;
//         {
//             std::unique_lock<std::mutex> lock(mtx);
//             cv.wait(lock, [] { return !task_part_queue[0].empty() || done; });

//             if (done && task_part_queue[0].empty())
//                 break;

//             task = task_part_queue[0].front();
//             task_part_queue[0].pop();
//         }

//         std::string filename = std::get<0>(task);
//         std::vector<float>& imageData = std::get<1>(task);
//         int& width = std::get<2>(task);
//         int& height = std::get<3>(task);
//         int& channels = std::get<4>(task);

//         decodeJPEG(filename, imageData, width, height, channels);
//         // Update task state and notify other threads
//         {
//             std::lock_guard<std::mutex> lock(mtx);
//             task_part_queue[1].push(task); // Queue for part 2
//         }
//         cv.notify_all();  // Notify all worker threads
//     }
// }

// // Worker function for part 2
// void worker_inference1(int cpu_id) {

//     while (true) {
//         std::tuple<std::string, std::vector<float>, int, int, int,int> task;
//         {   
//             std::unique_lock<std::mutex> lock(mtx);
//             cv.wait(lock, [] { return !task_part_queue[1].empty() || done; });

//             if (done && task_part_queue[1].empty())
//                 break;

//             task = task_part_queue[1].front();
//             task_part_queue[1].pop();
//         }

//         std::vector<float>& imageData = std::get<1>(task);
//         int& width = std::get<2>(task);
//         int& height = std::get<3>(task);
//         int& channels = std::get<4>(task);
//         int& id = std::get<5>(task);
//         inferenceTask(imageData, width, height, channels,id);

//         // Update task state and notify other threads
//         {
//             std::lock_guard<std::mutex> lock(mtx);
//             task_part_queue[2].push(task); // Queue for part 3
//         }
//         cv.notify_all();  // Notify all worker threads
//     }
// }

// // Worker function for part 3
// void worker_inference2(int cpu_id) {
//     while (true) {
//         std::tuple<std::string, std::vector<float>, int, int, int,int> task;
//         {
//             std::unique_lock<std::mutex> lock(mtx);
//             cv.wait(lock, [] { return !task_part_queue[2].empty() || done; });

//             if (done && task_part_queue[2].empty())
//                 break;

//             task = task_part_queue[2].front();
            
//             task_part_queue[2].pop();
//         }
//         int& id = std::get<5>(task);
//         inferenceTask2(id);
//     }
// }

// int main(int argc, char* argv[]) {
//     if (argc < 1) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }

//     // Start worker threads for each part
//     std::vector<std::thread> workers;
//     workers.emplace_back(worker_decode, 1);    // CPU 1 for decode
//     workers.emplace_back(worker_inference1, 2); // CPU 2 for inferenceTask
//     workers.emplace_back(worker_inference2, 3); // CPU 3 for inferenceTask2

//     //inter 1
//     std::string modelPath = "R-50-CG.tflite";
//     // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model =
//     tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     builder(&interpreter);
//     // 创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }

//     if (!interpreter) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter->AllocateTensors();
//     interpreter->SetNumThreads(4);
//     // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model2 =
//     tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver2;
//     tflite::InterpreterBuilder builder2(*model2, resolver2);
//     builder2(&interpreter2);
//     // 创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options2 = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate2 = TfLiteGpuDelegateV2Create(&options2);
//     if (interpreter2->ModifyGraphWithDelegate(delegate2) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate2);
//         return -1;
//     }

//     if (!interpreter2) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter2->AllocateTensors();
//     interpreter2->SetNumThreads(4);
//     // Push tasks to queue
//     char* files[8] = {
//         "1_jiqixuexi_1000.jpeg", "1_jiqixuexi_2000.jpeg", "1_jiqixuexi_3000.jpeg",
//         "1_jiqixuexi_4000.jpeg", "1_jiqixuexi_5000.jpeg", "1_jiqixuexi_6000.jpeg",
//         "1_jiqixuexi_7000.jpeg", "1_jiqixuexi_8000.jpeg"
//     };

//     for (int i = 0; i < 8; i++) {
//         std::string filename = files[i];
//         std::vector<float> imageData;
//         int width, height, channels;
//         {
//             std::lock_guard<std::mutex> lock(mtx);
//             task_part_queue[0].push(std::make_tuple(filename, imageData, width, height, channels, i));
//         }
//         cv.notify_all();  // Notify worker threads
//     }

//     // Signal threads to stop when done
//     {
//         //std::lock_guard<std::mutex> lock(mtx);
//         std::unique_lock<std::mutex> lock(mtx);
//         cv.wait(lock, [] {
//             return task_part_queue[0].empty() && task_part_queue[1].empty() && task_part_queue[2].empty();
//         });
//         done = true;
//     }
//     cv.notify_all();

//     for (auto& worker : workers) {
//         worker.join();
//     }
//     return 0;
// }


/*----------------------------------DCG- UINT8------------------*/
// #include <thread>
// #include <sched.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <jpeglib.h>
// #include <future>
// #include <mutex>
// #include <condition_variable>
// #include <unordered_map>
// #include <atomic>
// #include <queue>
// #include <functional>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "ctpl_stl.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"
// #include <cstdlib>
// #include <cstring>
// #include <stdexcept>
// #include <stdio.h>
// #include <unistd.h>
// #include <sys/syscall.h>
// #include <pthread.h>
// #include "jpegint.h"
// struct band_info {
//   short *dct;
//   unsigned int dct_h;
//   unsigned int dct_w;
//   unsigned int dct_b;
// };

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::unique_ptr<tflite::Interpreter> interpreter2;
// std::mutex mtx1;
// std::mutex mtx2;
// std::mutex mtx3;
// std::mutex mtx4;
// std::condition_variable cv;
// std::queue<std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *>> task_part_queue[4]; // Task queue for each part
// std::unordered_map<int, int> task_state; 
// bool done = false;
// int end_task1 = 2;
// int end_task2 = 3;
// int end_task2_1 = 2;
// int remaining_tasks = 0;
// int flag1 = 15000;
// int flag2 = 28000;

// void unpack_band_info(band_info* band, short **band_dct, int *band_dct_h,
//                       int *band_dct_w, int *band_dct_b) {
//   *band_dct = band->dct;
//   *band_dct_h = band->dct_h;
//   *band_dct_w = band->dct_w;
//   *band_dct_b = band->dct_b;
// }

// void dummy_dct_coefficients(band_info *band) {
//   band->dct_h = 0;
//   band->dct_w = 0;
//   band->dct_b = 0;
//   band->dct = new short[0];
// }
// void read_dct_coefficients(jpeg_decompress_struct *srcinfo,
//                            short *inputBuffer, int compNum,
//                            band_info *band, bool normalized,unsigned int totalBlock,int flag) {
//   if (compNum >= srcinfo->num_components) {// 检查通道索引是否合法
//     // make an empty component which would be half size of chroma // 创建一个空的组件，大小为色度的一半
//     band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2; //一个block高
//     band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;//一个block宽
//     band->dct_b = DCTSIZE2;//  可以调节输出通道数 每个block大小
//     long nb_elements =
//         (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);//有多上个元素
//     band->dct = new short[nb_elements];// 分配内存空间并初始化为 0
//     std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
//     return;
//   }

//   // prepare memory space dimensions // 准备内存空间的维度
//   // band->dct_h = srcinfo->comp_info[compNum].height_in_blocks;
//   // band->dct_w = srcinfo->comp_info[compNum].width_in_blocks;
//   // band->dct_b = DCTSIZE2;
//   long nb_elements =
//       (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);
//   // band->dct = new short[nb_elements];// 分配内存空间

//   int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
//   const unsigned short* quantval = normalized ? srcinfo->quant_tbl_ptrs[quant_idx]->quantval : nullptr;
//   short unscale = 1;

//   short *current_dct_coeff = band->dct;
//   short *current_dct_coeff2 = band->dct+(nb_elements/2);
//   if(compNum==0){
//       if(flag==0){
//       for(int i =0; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//                 if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                   unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//                 }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//               //std::cout<<"-----------"<<std::endl;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+5;
//     }
//       }
//     else{
//     for(int i =2; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//         if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//           unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//         }
//         *current_dct_coeff2=src[j]*unscale;
//         current_dct_coeff2++;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//         if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//           unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//         }
//         *current_dct_coeff2=src[j]*unscale;
//         current_dct_coeff2++;
//         }
//       i=i+5;
//     }
//     }
//   }
//   else{
//       for(int i =compNum+3; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//         for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+6;
//     }
//   }
// }

// void add_error_handler(jpeg_error_mgr *jerr) {
//   jerr->error_exit = [](j_common_ptr cinfo) {
//     char pszErr[1024];
//     (cinfo->err->format_message)(cinfo, pszErr);
//     fprintf(stderr, "Error: %s\n", pszErr);
//     exit(EXIT_FAILURE);
//   };
// }


// void transcode(jpeg_decompress_struct *srcinfo, unsigned char **outbuffer) {
//   static bool warning_emitted = false;
//   if (!warning_emitted) {// 发出警告，提示遇到非标准的 JPEG 图像，需要转码
//     fprintf(stderr, "WARNING: Non-standard JPEG image encountered, transcoding "
//                     "to H2_V2 which may negatively impact performance. This is "
//                     "the only time this warning will be shown.\n");
//     warning_emitted = true;
//   }

//   // start decompress
//   (void)jpeg_start_decompress(srcinfo);

//   // create the compression structure
//   jpeg_compress_struct dstinfo;
//   jpeg_error_mgr jerr;
//   dstinfo.err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_compress(&dstinfo);

//   size_t outlen = 0;
//   jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

//   dstinfo.image_width = srcinfo->image_width;
//   dstinfo.image_height = srcinfo->image_height;
//   dstinfo.input_components = srcinfo->output_components;
//   dstinfo.in_color_space = srcinfo->out_color_space;

//   jpeg_set_defaults(&dstinfo);
//   jpeg_set_quality(&dstinfo, 100, TRUE);
//   jpeg_start_compress(&dstinfo, TRUE);

//   // transcode// 转码  读出来立马写回去
//   unsigned char *line_buffer =
//       new unsigned char[srcinfo->output_width * srcinfo->output_components];
//   while (srcinfo->output_scanline < srcinfo->output_height) {
//     jpeg_read_scanlines(srcinfo, &line_buffer, 1);
//     (void)jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
//   }
//   delete[] line_buffer;
//   jpeg_finish_compress(&dstinfo);
//   jpeg_destroy_compress(&dstinfo);

//   // re-create decompress
//   jpeg_destroy_decompress(srcinfo);
//   jpeg_create_decompress(srcinfo);
//   jpeg_mem_src(srcinfo, *outbuffer, outlen);
//   (void)jpeg_read_header(srcinfo, TRUE);
// }

// bool is_grayscale(jpeg_decompress_struct *srcinfo) {
//   return srcinfo->num_components == 1;
// }

// bool is_h2_v2(jpeg_decompress_struct *srcinfo) {
//   return (srcinfo->comp_info[0].h_samp_factor == 2) &&
//          (srcinfo->comp_info[0].v_samp_factor == 2) &&
//          (srcinfo->comp_info[1].h_samp_factor == 1) &&
//          (srcinfo->comp_info[1].v_samp_factor == 1) &&
//          (srcinfo->comp_info[2].h_samp_factor == 1) &&
//          (srcinfo->comp_info[2].v_samp_factor == 1);
// }

// struct thread_data {
//     int start;
//     int end;
//     j_decompress_ptr info;
//     JBLOCKROW* mcu_buffer;
//     JBLOCKROW block_buffer;
//     int thread;
//     int line;
// };
// struct result_data {
//     int num;
//     j_decompress_ptr info;
//     short * inputBuffer;
//     int thread;
//     int normalized;
//     unsigned int totalBlock;
//     band_info * band;
//     int flag;
// };
// struct result_data2 {
//     j_decompress_ptr info;
//     short * inputBuffer;
//     int thread;
//     int normalized;
//     unsigned int totalBlock;
//     band_info * band;
//     band_info * band2;
//     int flag;
// };

// // 这个函数处理从start到end的MCU
// void* process_mcus(void* args) {

//     struct thread_data* data = (struct thread_data*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     for (int i = data->start; i < data->end; ++i) {
//         data->info->line[data->line] = i;
//         for (int j = 0; j < data->info->blocks_in_MCU; ++j) {
//             data->mcu_buffer[j] = data->block_buffer + data->info->blocks_in_MCU * i + j;
//         }
//         if (FALSE == (*data->info->entropy->decode_mcu_mult)(data->info, data->mcu_buffer, data->line)) {
//             break;
//         }
//     }
//     return NULL;
// }

// void* process_result(void* args) {

//     struct result_data* data = (struct result_data*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     read_dct_coefficients(data->info, data->inputBuffer, data->num, data->band, data->normalized,data->totalBlock,data->flag);
//     return NULL;
// }
// void* process_result2(void* args) {

//     struct result_data2* data = (struct result_data2*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     read_dct_coefficients(data->info, data->inputBuffer, 1, data->band, data->normalized,data->totalBlock,data->flag);
//     read_dct_coefficients(data->info, data->inputBuffer, 2, data->band2, data->normalized,data->totalBlock,data->flag);
//     return NULL;
// }
// void read_dct_coefficients_from_srcinfo_mult(jpeg_decompress_struct *srcinfo,
//                                         bool normalized, int channels,
//                                         band_info *band1, band_info *band2,
//                                         band_info *band3,unsigned char *buffer){
//   unsigned int totalMcuNum = srcinfo->MCU_rows_in_scan * srcinfo->MCUs_per_row;
//   unsigned int totalBlock = srcinfo->blocks_in_MCU * totalMcuNum;
//     short *inputBuffer = (short*)malloc(totalBlock * sizeof(JBLOCK));
//   JBLOCK **mcu_buffer = (JBLOCK**)malloc(sizeof(JBLOCK*) * srcinfo->blocks_in_MCU);
//   JBLOCK *block_buffer = (JBLOCK*)inputBuffer;
//   pthread_t thread1, thread2,thread3,thread4;
//   struct thread_data t1_data = {0, flag1 , srcinfo, mcu_buffer, block_buffer,7,1};
//   struct thread_data t2_data = {flag1, flag2, srcinfo, mcu_buffer, block_buffer,6,2};
//   struct thread_data t3_data = {flag2, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,5,3};
//   //struct thread_data t4_data = {31900, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,4,4};
//   srcinfo->unread_marker = 0;
//   srcinfo->unread_marker2 = 0;
//   srcinfo->unread_marker3 = 0;
//   srcinfo->now_b1_id=1;
//   srcinfo->now_b2_id=srcinfo->huff_scan.buffer_id[flag1]-1;//di er ge xian cheng kai shi de buffer id;
//   srcinfo->now_b3_id=srcinfo->huff_scan.buffer_id[flag2]-1;
//   //srcinfo->now_b4_id=srcinfo->huff_scan.buffer_id[31900]-1;
//   (*srcinfo->src->retrun_file_start) (srcinfo);
//   // 创建线程1
//   pthread_create(&thread1, NULL, process_mcus, &t1_data);
//   // 创建线程2
//   pthread_create(&thread2, NULL, process_mcus, &t2_data);
//   // 创建线程1
//   pthread_create(&thread3, NULL, process_mcus, &t3_data);
// //   // 创建线程2
// //   pthread_create(&thread4, NULL, process_mcus, &t4_data);

//   // 等待两个线程完成
//   pthread_join(thread1, NULL);

//   pthread_join(thread2, NULL);

//   pthread_join(thread3, NULL);
//       //usleep(50000);
//   pthread_join(thread4, NULL);
//     band1->dct_h = srcinfo->comp_info[0].height_in_blocks;
//   band1->dct_w = srcinfo->comp_info[0].width_in_blocks;
//   band1->dct_b = DCTSIZE2;
//   long nb_elements1 =
//       (long)(band1->dct_h) * (long)(band1->dct_w) * (long)(band1->dct_b);
//   band1->dct = new short[nb_elements1];// 分配内存空间
//       band2->dct_h = srcinfo->comp_info[1].height_in_blocks;
//   band2->dct_w = srcinfo->comp_info[1].width_in_blocks;
//   band2->dct_b = DCTSIZE2;
//   long nb_elements2 =
//       (long)(band2->dct_h) * (long)(band2->dct_w) * (long)(band2->dct_b);
//   band2->dct = new short[nb_elements2];// 分配内存空间
//       band3->dct_h = srcinfo->comp_info[2].height_in_blocks;
//   band3->dct_w = srcinfo->comp_info[2].width_in_blocks;
//   band3->dct_b = DCTSIZE2;
//   long nb_elements3 =
//       (long)(band3->dct_h) * (long)(band3->dct_w) * (long)(band3->dct_b);
//   band3->dct = new short[nb_elements3];// 分配内存空间
//   auto start = std::chrono::high_resolution_clock::now();
//   struct result_data r1 = {0,srcinfo,inputBuffer,7,normalized,totalBlock,band1,0};
//   struct result_data r3 = {0,srcinfo,inputBuffer,6,normalized,totalBlock,band1,1};
//   //struct result_data r2 = {1,srcinfo,inputBuffer,6,normalized,totalBlock,band2};
//   //struct result_data r3 = {2,srcinfo,inputBuffer,5,normalized,totalBlock,band3};
//   struct result_data2 r2 = {srcinfo,inputBuffer,5,normalized,totalBlock,band2,band3,0};
//   // auto start_ii = std::chrono::high_resolution_clock::now();
//   // for (int ii=0; ii<10;ii++){
//   pthread_create(&thread1, NULL, process_result, &r1);
//   //pthread_create(&thread2, NULL, process_result, &r2);
//   pthread_create(&thread3, NULL, process_result, &r3);
//   // 创建线程2
//   pthread_create(&thread2, NULL, process_result2, &r2);

//   pthread_join(thread1, NULL);

//   pthread_join(thread2, NULL);
//   pthread_join(thread3, NULL);
// //   }
// // auto end_ii = std::chrono::high_resolution_clock::now();
// // auto duration_invoke_ii = std::chrono::duration_cast<std::chrono::milliseconds>(end_ii - start_ii);
// //   std::cout<<"copy:"<<duration_invoke_ii.count()<<std::endl;
//   // read_dct_coefficients(srcinfo, inputBuffer, 0, band1, normalized,totalBlock);// 读取第一个通道的 DCT 系数
//   // if (channels == 3) {// 如果通道数为 3，则继续读取第二个和第三个通道的 DCT 系数，否则将其设置为虚拟的 DCT 系数
//   // read_dct_coefficients(srcinfo, inputBuffer, 1, band2, normalized,totalBlock);
//   // read_dct_coefficients(srcinfo, inputBuffer, 2, band3, normalized,totalBlock);
//   // } else {
//   //   dummy_dct_coefficients(band2);
//   //   dummy_dct_coefficients(band3);
//   // }
//    // auto end = std::chrono::high_resolution_clock::now();
//   //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   //std::cout<<duration.count()<<std::endl;
//     //usleep(20000);
//   if (buffer != nullptr) {
//     delete[] buffer;
//   }
// }
// void read_dct_coefficients_from_file_scan(char *filename,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
  
//   if ((infile = fopen(filename, "rb")) == nullptr) {
//     fprintf(stderr, "ERROR: can't open %s\n", filename);
//     return;
//   }

//   jpeg_error_mgr jerr;
//   srcinfo->err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_decompress(srcinfo);
//   jpeg_stdio_src(srcinfo, infile);
  
//   jpeg_read_header(srcinfo, TRUE);
//   jpeg_start_decompress(srcinfo);

//   if (!is_grayscale(srcinfo) && !is_h2_v2(srcinfo)) {// 如果图像不是灰度且不是 H2V2 格式，则进行转码
//     transcode(srcinfo, &buffer);
//   }
//   // 读取 JPEG 文件的 DCT 系数
//   jpeg_read_coefficients_scan(srcinfo);

// }
// void read_dct_coefficients_from_file_mult(char *filename, bool normalized,
//                                       int channels, band_info *band1,
//                                       band_info *band2, band_info *band3,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
//   read_dct_coefficients_from_srcinfo_mult(srcinfo,normalized,channels,band1,band2,band3,buffer);
//   jpeg_abort_decompress(srcinfo);
//   jpeg_destroy_decompress(srcinfo);
//   //fclose(infile);
// }

// // 示例函数，实现 short* 数据的 resize
// short* resizeShortData(const short* input_data, int original_height, int original_width, int channels,
//                        int new_height, int new_width) {
//     // 动态分配内存给 resized_data
//     short* resized_data = new short[new_height * new_width * channels];

//     // 计算缩放比例
//     float scale_h = static_cast<float>(original_height - 1) / (new_height - 1);
//     float scale_w = static_cast<float>(original_width - 1) / (new_width - 1);

//     // 对每个像素进行插值
//     for (int y = 0; y < new_height; ++y) {
//         for (int x = 0; x < new_width; ++x) {
//             float gy = y * scale_h;
//             float gx = x * scale_w;

//             int gxi = static_cast<int>(gx);
//             int gyi = static_cast<int>(gy);

//             for (int c = 0; c < channels; ++c) {
//                 float c00 = input_data[(gyi * original_width + gxi) * channels + c];
//                 float c10 = input_data[(gyi * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];
//                 float c01 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + gxi) * channels + c];
//                 float c11 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];

//                 float cx0 = c00 + (c10 - c00) * (gx - gxi);
//                 float cx1 = c01 + (c11 - c01) * (gx - gxi);
//                 float cxy = cx0 + (cx1 - cx0) * (gy - gyi);

//                 resized_data[(y * new_width + x) * channels + c] = static_cast<short>(std::round(cxy));
//             }
//         }
//     }

//     return resized_data;
// }

// std::vector<int8_t> addBatchDimension(const std::vector<int8_t>& imageData, int height, int width, int channels) {
//     std::vector<int8_t> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// std::vector<int8_t> convertToInt8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<int8_t> int8_vector;
//     long nb_elements = static_cast<long>(dct_h) * static_cast<long>(dct_w) * static_cast<long>(dct_b);
//     int8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         int8_vector.push_back(static_cast<int8_t>(band_dct[i]));
//     }
//     return int8_vector;
// }
// void inferenceTask2(int id) {
//   if(id%2==0){
//     interpreter->Invoke_2();
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     const float* output_data_uint8 = output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1];
//   }
//   else{
//     interpreter2->Invoke_2();
//     int output_tensor_idx = interpreter2->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter2->tensor(output_tensor_idx);
//     const float* output_data_uint8 = output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1];
//   }
// }

// void inferenceTask(short* band1_dct, int band1_dct_h, int band1_dct_w, int band1_dct_b,
//                          short* band2_dct, int band2_dct_h, int band2_dct_w, int band2_dct_b,
//                          short* band3_dct, int band3_dct_h, int band3_dct_w, int band3_dct_b,int id) {

//     int int8_min = -128;
//     int int8_max = 127;
//     int min_val = -1024;
//     int max_val = 1016;
//     int target_size1=28;
//     int target_size2=14;
//     //usleep(50000);
//     short* resized_data1 = resizeShortData(band1_dct,band1_dct_h, band1_dct_w, band1_dct_b, target_size1, target_size1);

//     short* resized_data2 = resizeShortData(band2_dct,band2_dct_h, band2_dct_w, band2_dct_b, target_size2, target_size2);

//     short* resized_data3 = resizeShortData(band3_dct,band3_dct_h, band3_dct_w, band3_dct_b, target_size2, target_size2);

//     std::vector<int8_t> input_data1(target_size1 * target_size1 * band1_dct_b);
//     std::vector<int8_t> input_data2(target_size2 * target_size2 * band2_dct_b);
//     std::vector<int8_t> input_data3(target_size2 * target_size2 * band3_dct_b);
//     input_data1 = convertToInt8Vector(resized_data1,target_size1, target_size1, band1_dct_b);
//     input_data2 = convertToInt8Vector(resized_data2,target_size2, target_size2, band2_dct_b);
//     input_data3 = convertToInt8Vector(resized_data3,target_size2, target_size2, band3_dct_b);

//     std::vector<int8_t> input1_batch = addBatchDimension(input_data1,target_size1, target_size1, band1_dct_b);
//     std::vector<int8_t> input2_batch = addBatchDimension(input_data2,target_size2, target_size2, band2_dct_b);
//     std::vector<int8_t> input3_batch = addBatchDimension(input_data3,target_size2, target_size2, band3_dct_b);
//     //std::cout<<"Invoke-----------"<<id<<std::endl;
//     //usleep(20000);
//     if(id%2==0){

//     // 获取输入张量索引
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     int input_index3 = interpreter->inputs()[2];

//     // 确保输入张量的大小正确
//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     TfLiteTensor* tensor3 = interpreter->tensor(input_index3);


//     if (tensor1->bytes != input1_batch.size() ||
//         tensor2->bytes != input2_batch.size() ||
//         tensor3->bytes != input3_batch.size() ) {
//         std::cout<<tensor1->bytes<<"::"<<input1_batch.size()<<std::endl;
//         std::cout<<tensor2->bytes<<"::"<<input2_batch.size()<<std::endl;
//         std::cout<<tensor3->bytes<<"::"<<input3_batch.size()<<std::endl;
//         std::cerr << "Input tensor size does not match data size." << std::endl;
//         return;
//     }
//     std::memcpy(tensor1->data.int8, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.int8, input2_batch.data(), input2_batch.size());
//     std::memcpy(tensor3->data.int8, input3_batch.data(), input3_batch.size());
    
//     // 4. Run inference
//     interpreter->Invoke();
//     }
//     else{
//           // 获取输入张量索引
//     int input_index1 = interpreter2->inputs()[0];
//     int input_index2 = interpreter2->inputs()[1];
//     int input_index3 = interpreter2->inputs()[2];

//     // 确保输入张量的大小正确
//     TfLiteTensor* tensor1 = interpreter2->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter2->tensor(input_index2);
//     TfLiteTensor* tensor3 = interpreter2->tensor(input_index3);


//     if (tensor1->bytes != input1_batch.size() ||
//         tensor2->bytes != input2_batch.size() ||
//         tensor3->bytes != input3_batch.size() ) {
//         std::cout<<tensor1->bytes<<"::"<<input1_batch.size()<<std::endl;
//         std::cout<<tensor2->bytes<<"::"<<input2_batch.size()<<std::endl;
//         std::cout<<tensor3->bytes<<"::"<<input3_batch.size()<<std::endl;
//         std::cerr << "Input tensor size does not match data size." << std::endl;
//         return;
//     }
//     std::memcpy(tensor1->data.int8, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.int8, input2_batch.data(), input2_batch.size());
//     std::memcpy(tensor3->data.int8, input3_batch.data(), input3_batch.size());
    
//     // 4. Run inference
//     interpreter2->Invoke();
//     }
// }

// // Worker function for part 1 using read_dct_coefficients_from_file
// void worker_decode(int cpu_id) {
//     while (true) {
//         std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
//         {
//             std::unique_lock<std::mutex> lock(mtx1);
//             cv.wait(lock, [] { return (!task_part_queue[0].empty() && end_task1 && end_task2) || done; });
//             end_task1--;
//             end_task2--;
//             if (done && task_part_queue[0].empty())
//                 break;

//             task = task_part_queue[0].front();
//             task_part_queue[0].pop();
                              
//         }

//         // Extract parameters from tuple
//         char* filename = std::get<0>(task);
//         bool normalized = std::get<1>(task);
//         int channels = std::get<2>(task);
//         short **band1_dct = std::get<3>(task);
//         int *band1_dct_h = std::get<4>(task);
//         int *band1_dct_w = std::get<5>(task);
//         int *band1_dct_b = std::get<6>(task);
//         short **band2_dct = std::get<7>(task);
//         int *band2_dct_h = std::get<8>(task);
//         int *band2_dct_w = std::get<9>(task);
//         int *band2_dct_b = std::get<10>(task);
//         short **band3_dct = std::get<11>(task);
//         int *band3_dct_h = std::get<12>(task);
//         int *band3_dct_w = std::get<13>(task);
//         int *band3_dct_b = std::get<14>(task);
//         int id = std::get<15>(task);
//         FILE *infile = std::get<16>(task);
//         band_info *band1 = std::get<17>(task);
//         band_info *band2 = std::get<18>(task);
//         band_info *band3 = std::get<19>(task);
//         jpeg_decompress_struct *srcinfo = std::get<20>(task); 
//         unsigned char *buffer = std::get<21>(task);
//         //std::cout<<"JPEG-----------"<<id<<std::endl;
//  	    cpu_set_t mask;
//       CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
//       CPU_SET(4, &mask);
//       if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
//        }
//         auto start_1 = std::chrono::high_resolution_clock::now();
//         // Call the DCT read function
//         read_dct_coefficients_from_file_scan(filename,infile,srcinfo,buffer);

//        auto end_1 = std::chrono::high_resolution_clock::now();
//        std::cout<<"stage1:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count()<<std::endl;
//         // Update task state and notify other threads
//         {
//             std::lock_guard<std::mutex> lock(mtx1);
//             task_part_queue[1].push(task);  // Queue forpart 2
//         }
//         cv.notify_all();  // Notify all worker threads
//     }
// }


// // Worker function for part 2
// void worker_inference1(int cpu_id) {

//     while (true) {
//         std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
//         {   
//             std::unique_lock<std::mutex> lock(mtx2);
//             cv.wait(lock, [] { return (!task_part_queue[1].empty() && end_task2_1) || done; });
//             end_task2_1--;
//             if (done && task_part_queue[1].empty())
//                 break;

//             task = task_part_queue[1].front();
//             task_part_queue[1].pop();
//         }

//         char* filename = std::get<0>(task);
//         bool normalized = std::get<1>(task);
//         int channels = std::get<2>(task);
//         short **band1_dct = std::get<3>(task);
//         int *band1_dct_h = std::get<4>(task);
//         int *band1_dct_w = std::get<5>(task);
//         int *band1_dct_b = std::get<6>(task);
//         short **band2_dct = std::get<7>(task);
//         int *band2_dct_h = std::get<8>(task);
//         int *band2_dct_w = std::get<9>(task);
//         int *band2_dct_b = std::get<10>(task);
//         short **band3_dct = std::get<11>(task);
//         int *band3_dct_h = std::get<12>(task);
//         int *band3_dct_w = std::get<13>(task);
//         int *band3_dct_b = std::get<14>(task);
//         int id = std::get<15>(task);
//         FILE *infile = std::get<16>(task);
//         band_info *band1 = std::get<17>(task);
//         band_info *band2 = std::get<18>(task);
//         band_info *band3 = std::get<19>(task);
//         jpeg_decompress_struct *srcinfo = std::get<20>(task); 
//         unsigned char *buffer = std::get<21>(task);
//          auto start_2 = std::chrono::high_resolution_clock::now();
//         read_dct_coefficients_from_file_mult(filename,normalized,channels,band1,band2,band3,infile,srcinfo,buffer);
//         unpack_band_info(band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
//         unpack_band_info(band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
//         unpack_band_info(band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
// //std::cout<<"huffman end---------------"<<std::endl;
// // auto end_1 = std::chrono::high_resolution_clock::now();
// // std::cout<<"huff2:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_2).count()<<std::endl;
//         inferenceTask(*band1_dct, *band1_dct_h, *band1_dct_w, *band1_dct_b,*band2_dct, *band2_dct_h, *band2_dct_w, *band2_dct_b,*band3_dct, *band3_dct_h, *band3_dct_w, *band3_dct_b,id);
//         //  auto end_2 = std::chrono::high_resolution_clock::now();
//         //  std::cout<<"stage2:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - end_1).count()<<std::endl;
//         // Update task state and notify other threads
//         {
//             std::lock_guard<std::mutex> lock(mtx2);
//             task_part_queue[2].push(task); // Queue for part 3
//             end_task1++;
            
//         }
//         cv.notify_all();  // Notify all worker threads
//     }
// }

// // Worker function for part 3
// void worker_inference2(int cpu_id) {
//     while (true) {
//         std::tuple<char*, bool, int, short**, int*, int*, int*, short**, int*, int*, int*, short**, int*, int*, int*, int , FILE *, band_info*, band_info*, band_info*,jpeg_decompress_struct*, unsigned char *> task;
//         {
//             std::unique_lock<std::mutex> lock(mtx3);
//             cv.wait(lock, [] { return !task_part_queue[2].empty() || done; });
//             if (done && task_part_queue[2].empty())
//                 break;

//             task = task_part_queue[2].front();
//             task_part_queue[2].pop();
//         }
//         int id = std::get<15>(task);
//          //auto start_3 = std::chrono::high_resolution_clock::now();
//         //inferenceTask2(id);

//         // auto end_3 = std::chrono::high_resolution_clock::now();
//         // std::cout<<"stage3:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end_3 - start_3).count()<<std::endl;
//         {
//             std::lock_guard<std::mutex> lock(mtx3);
//             end_task2++;
//             end_task2_1++;
//             --remaining_tasks;
//         }

//         cv.notify_all();
        
//     }
// }

// int main(int argc, char* argv[]){
//     if (argc < 3) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//       //cpu_set_t mask;
//       //CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
//       //CPU_SET(, &mask);
//       //if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
//       // }
//     flag1 = std::stoi(argv[1]);
//     flag2 = std::stoi(argv[2]);
//     std::string modelPath = "DCG76ACC.tflite";
//     // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model =
//     tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*model, resolver);
//     builder(&interpreter);
//     // 创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }

//     if (!interpreter) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter->AllocateTensors();
//     interpreter->SetNumThreads(3);
//     // 替换为你的预处理后图像文件路径和TFLite模型文件路径
//     // 2. Initialize TensorFlow Lite interpreter
//     std::unique_ptr<tflite::FlatBufferModel> model2 =
//     tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//     tflite::ops::builtin::BuiltinOpResolver resolver2;
//     tflite::InterpreterBuilder builder2(*model2, resolver2);
//     builder2(&interpreter2);
//     // 创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options2 = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate2 = TfLiteGpuDelegateV2Create(&options2);
//     if (interpreter2->ModifyGraphWithDelegate(delegate2) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate2);
//         return -1;
//     }

//     if (!interpreter2) {
//         std::cerr << "Failed to create interpreter." << std::endl;
//         return -1;
//     }

//     // 3. Allocate tensors and set inputs
//     interpreter2->AllocateTensors();
//     interpreter2->SetNumThreads(3);

//         char* files[8] = {
//         "83.jpeg", "84.jpeg", "85.jpeg",
//         "86.jpeg", "87.jpeg", "88.jpeg",
//         "89.jpeg", "90.jpeg"
//     };

//     // Start worker threads for each part
//     std::vector<std::thread> workers;
//     workers.emplace_back(worker_decode, 1);    // CPU 1 for decode
//     workers.emplace_back(worker_inference1, 2); // CPU 2 for inferenceTask
//     workers.emplace_back(worker_inference2, 3); // CPU 3 for inferenceTask2
//     remaining_tasks = 50;
//     int size = 1000*1000*64;
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < remaining_tasks; i++) {// jiang suo you ren wu jia ru dao dui lei zhong 
//         char* filename = files[i%8];
//         short* band1_dct = new short[size];
//         int* band1_dct_h = new int(0);
//         int* band1_dct_w = new int(0);
//         int* band1_dct_b = new int(0);
//         short* band2_dct = new short[size];
//         int* band2_dct_h = new int(0);
//         int* band2_dct_w = new int(0);
//         int* band2_dct_b = new int(0);
//         short* band3_dct = new short[size];
//         int* band3_dct_h = new int(0);
//         int* band3_dct_w = new int(0);
//         int* band3_dct_b = new int(0);
//         bool normalized = true;  // Example value
//         int channels = 3;        // Example value
//         FILE* infile = nullptr;
//         band_info* band1 = new band_info();
//         band_info* band2 = new band_info();
//         band_info* band3 = new band_info();
//         jpeg_decompress_struct* srcinfo = new jpeg_decompress_struct();
//         unsigned char* buffer = nullptr;
//         {
//           std::lock_guard<std::mutex> lock(mtx1);
//                     task_part_queue[0].push(std::make_tuple(filename, normalized, channels, 
//                                                 static_cast<short**>(&band1_dct), band1_dct_h, band1_dct_w, band1_dct_b,
//                                                 static_cast<short**>(&band2_dct), band2_dct_h, band2_dct_w, band2_dct_b,
//                                                 static_cast<short**>(&band3_dct), band3_dct_h, band3_dct_w, band3_dct_b,
//                                                 i, infile, band1, band2, band3, srcinfo, buffer));
//     }
//         cv.notify_all();  // Notify worker threads
//     }
//     // Signal threads to stop when done
//     {
//         //std::lock_guard<std::mutex> lock(mtx);
//         std::unique_lock<std::mutex> lock(mtx4);
//         cv.wait(lock, [] {
//              return remaining_tasks == 0;
//         });
//         done = true;
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     cv.notify_all();
//     for (auto& worker : workers) {
//         worker.join();
//     }
//   auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   std::cout<<"all:"<<duration_invoke.count()<<std::endl;
// 	return 0;
// }



/*----------------------------duo he haffuman DCT-DCG-danzhang--------------------*/
// #include <thread>
// #include <sched.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <jpeglib.h>
// #include <future>
// #include <mutex>
// #include <condition_variable>
// #include <unordered_map>
// #include <atomic>
// #include <queue>
// #include <functional>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "ctpl_stl.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"
// #include <cstdlib>
// #include <cstring>
// #include <stdexcept>
// #include <stdio.h>
// #include <unistd.h>
// #include <sys/syscall.h>
// #include <pthread.h>
// #include "jpegint.h"
// struct band_info {
//   short *dct;
//   unsigned int dct_h;
//   unsigned int dct_w;
//   unsigned int dct_b;
// };

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// //using namespace jpeg2dct::common;
// std::vector<float> convertToUint8Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<float> uint8_vector;
//     long nb_elements = (long)(dct_h) * (long)(dct_w) * (long)(dct_b);
//     uint8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         uint8_vector.push_back(static_cast<float>(band_dct[i]));
//     }
//     return uint8_vector;
// }

// // Updated resizeImage function for NHWC format
// short* resizeShortData(const short* input_data, int original_height, int original_width, int channels,
//                        int new_height, int new_width) {
//     // 动态分配内存给 resized_data
//     short* resized_data = new short[new_height * new_width * channels];

//     // 计算缩放比例
//     float scale_h = static_cast<float>(original_height - 1) / (new_height - 1);
//     float scale_w = static_cast<float>(original_width - 1) / (new_width - 1);

//     // 对每个像素进行插值
//     for (int y = 0; y < new_height; ++y) {
//         for (int x = 0; x < new_width; ++x) {
//             float gy = y * scale_h;
//             float gx = x * scale_w;

//             int gxi = static_cast<int>(gx);
//             int gyi = static_cast<int>(gy);

//             for (int c = 0; c < channels; ++c) {
//                 float c00 = input_data[(gyi * original_width + gxi) * channels + c];
//                 float c10 = input_data[(gyi * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];
//                 float c01 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + gxi) * channels + c];
//                 float c11 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];

//                 float cx0 = c00 + (c10 - c00) * (gx - gxi);
//                 float cx1 = c01 + (c11 - c01) * (gx - gxi);
//                 float cxy = cx0 + (cx1 - cx0) * (gy - gyi);

//                 resized_data[(y * new_width + x) * channels + c] = static_cast<short>(std::round(cxy));
//             }
//         }
//     }

//     return resized_data;
// }

// const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

// const int height = 14;
// const int width = 14;
// const int depth1 = 64;  // 第一个输入的通道数
// const int depth2 = 64;  // 第二个输入的通道数
// // 合并函数，将两个 14x14x64 的输入合并为一个 14x14x128
// std::vector<float> mergeInputs(const std::vector<float>& input1, const std::vector<float>& input2) {
//     // 检查输入大小是否匹配
//     if (input1.size() != height * width * depth1 || input2.size() != height * width * depth2) {
//         std::cerr << "输入大小不匹配!" << std::endl;
//         return {};
//     }

//     // 创建合并后的 vector，大小为14x14x128
//     std::vector<float> merged_input(height * width * (depth1 + depth2));

//     // 逐元素合并
//     for (int i = 0; i < height * width; ++i) {
//         // 将 input1 的数据拷贝到 merged_input 中
//         for (int d1 = 0; d1 < depth1; ++d1) {
//             merged_input[i * (depth1 + depth2) + d1] = input1[i * depth1 + d1];
//         }
//         // 将 input2 的数据拷贝到 merged_input 中
//         for (int d2 = 0; d2 < depth2; ++d2) {
//             merged_input[i * (depth1 + depth2) + depth1 + d2] = input2[i * depth2 + d2];
//         }
//     }

//     return merged_input;
// }

// int flag1 = 18000;
// int flag2 = 25500;
// int flag3 = 33000;

// void unpack_band_info(band_info* band, short **band_dct, int *band_dct_h,
//                       int *band_dct_w, int *band_dct_b) {
//   *band_dct = band->dct;
//   *band_dct_h = band->dct_h;
//   *band_dct_w = band->dct_w;
//   *band_dct_b = band->dct_b;
// }

// void dummy_dct_coefficients(band_info *band) {
//   band->dct_h = 0;
//   band->dct_w = 0;
//   band->dct_b = 0;
//   band->dct = new short[0];
// }
// void read_dct_coefficients(jpeg_decompress_struct *srcinfo,
//                            short *inputBuffer, int compNum,
//                            band_info *band, bool normalized,unsigned int totalBlock,int flag) {
//   if (compNum >= srcinfo->num_components) {// 检查通道索引是否合法
//     // make an empty component which would be half size of chroma // 创建一个空的组件，大小为色度的一半
//     band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2; //一个block高
//     band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;//一个block宽
//     band->dct_b = DCTSIZE2;//  可以调节输出通道数 每个block大小
//     long nb_elements =
//         (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);//有多上个元素
//     band->dct = new short[nb_elements];// 分配内存空间并初始化为 0
//     std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
//     return;
//   }


//   long nb_elements =
//       (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);

//   int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
//   const unsigned short* quantval = normalized ? srcinfo->quant_tbl_ptrs[quant_idx]->quantval : nullptr;
//   short unscale = 1;

//   short *current_dct_coeff = band->dct;
//   short *current_dct_coeff2 = band->dct+(nb_elements/2);
//   if(compNum==0){
//       if(flag==0){
//       for(int i =0; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//                 if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                   unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//                 }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+5;
//     }
//       }
//     else{
//     for(int i =2; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//         if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//           unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//         }
//         *current_dct_coeff2=src[j]*unscale;
//         current_dct_coeff2++;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//         if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//           unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//         }
//         *current_dct_coeff2=src[j]*unscale;
//         current_dct_coeff2++;
//         }
//       i=i+5;
//     }
//     }
//   }
//   else{
//       for(int i =compNum+3; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//         for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+6;
//     }
//   }
// }

// void add_error_handler(jpeg_error_mgr *jerr) {
//   jerr->error_exit = [](j_common_ptr cinfo) {
//     char pszErr[1024];
//     (cinfo->err->format_message)(cinfo, pszErr);
//     fprintf(stderr, "Error: %s\n", pszErr);
//     exit(EXIT_FAILURE);
//   };
// }


// void transcode(jpeg_decompress_struct *srcinfo, unsigned char **outbuffer) {
//   static bool warning_emitted = false;
//   if (!warning_emitted) {// 发出警告，提示遇到非标准的 JPEG 图像，需要转码
//     fprintf(stderr, "WARNING: Non-standard JPEG image encountered, transcoding "
//                     "to H2_V2 which may negatively impact performance. This is "
//                     "the only time this warning will be shown.\n");
//     warning_emitted = true;
//   }

//   // start decompress
//   (void)jpeg_start_decompress(srcinfo);

//   // create the compression structure
//   jpeg_compress_struct dstinfo;
//   jpeg_error_mgr jerr;
//   dstinfo.err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_compress(&dstinfo);

//   size_t outlen = 0;
//   jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

//   dstinfo.image_width = srcinfo->image_width;
//   dstinfo.image_height = srcinfo->image_height;
//   dstinfo.input_components = srcinfo->output_components;
//   dstinfo.in_color_space = srcinfo->out_color_space;

//   jpeg_set_defaults(&dstinfo);
//   jpeg_set_quality(&dstinfo, 100, TRUE);
//   jpeg_start_compress(&dstinfo, TRUE);

//   // transcode// 转码  读出来立马写回去
//   unsigned char *line_buffer =
//       new unsigned char[srcinfo->output_width * srcinfo->output_components];
//   while (srcinfo->output_scanline < srcinfo->output_height) {
//     jpeg_read_scanlines(srcinfo, &line_buffer, 1);
//     (void)jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
//   }
//   delete[] line_buffer;
//   jpeg_finish_compress(&dstinfo);
//   jpeg_destroy_compress(&dstinfo);

//   // re-create decompress
//   jpeg_destroy_decompress(srcinfo);
//   jpeg_create_decompress(srcinfo);
//   jpeg_mem_src(srcinfo, *outbuffer, outlen);
//   (void)jpeg_read_header(srcinfo, TRUE);
// }

// bool is_grayscale(jpeg_decompress_struct *srcinfo) {
//   return srcinfo->num_components == 1;
// }

// bool is_h2_v2(jpeg_decompress_struct *srcinfo) {
//   return (srcinfo->comp_info[0].h_samp_factor == 2) &&
//          (srcinfo->comp_info[0].v_samp_factor == 2) &&
//          (srcinfo->comp_info[1].h_samp_factor == 1) &&
//          (srcinfo->comp_info[1].v_samp_factor == 1) &&
//          (srcinfo->comp_info[2].h_samp_factor == 1) &&
//          (srcinfo->comp_info[2].v_samp_factor == 1);
// }

// struct thread_data {
//     int start;
//     int end;
//     j_decompress_ptr info;
//     JBLOCKROW* mcu_buffer;
//     JBLOCKROW block_buffer;
//     int thread;
//     int line;
// };
// struct result_data {
//     int num;
//     j_decompress_ptr info;
//     short * inputBuffer;
//     int thread;
//     int normalized;
//     unsigned int totalBlock;
//     band_info * band;
//     int flag;
// };
// struct result_data2 {
//     j_decompress_ptr info;
//     short * inputBuffer;
//     int thread;
//     int normalized;
//     unsigned int totalBlock;
//     band_info * band;
//     band_info * band2;
//     int flag;
// };

// // 这个函数处理从start到end的MCU
// void* process_mcus(void* args) {

//     struct thread_data* data = (struct thread_data*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     for (int i = data->start; i < data->end; ++i) {
//         data->info->line[data->line] = i;
//         for (int j = 0; j < data->info->blocks_in_MCU; ++j) {
//             data->mcu_buffer[j] = data->block_buffer + data->info->blocks_in_MCU * i + j;
//         }
//         if (FALSE == (*data->info->entropy->decode_mcu_mult)(data->info, data->mcu_buffer, data->line)) {
//             break;
//         }
//     }
//     return NULL;
// }

// void* process_result(void* args) {

//     struct result_data* data = (struct result_data*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     read_dct_coefficients(data->info, data->inputBuffer, data->num, data->band, data->normalized,data->totalBlock,data->flag);
//     return NULL;
// }
// void* process_result2(void* args) {

//     struct result_data2* data = (struct result_data2*)args;
//     // 获取当前线程的真实 ID
//     pid_t tid = syscall(SYS_gettid);

//     // 设置 CPU 亲和性
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     CPU_SET(data->thread, &cpuset); // 简单绑定到 CPU 核 0 或 核 1
//     sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);

//     read_dct_coefficients(data->info, data->inputBuffer, 1, data->band, data->normalized,data->totalBlock,data->flag);
//     read_dct_coefficients(data->info, data->inputBuffer, 2, data->band2, data->normalized,data->totalBlock,data->flag);
//     return NULL;
// }
// void read_dct_coefficients_from_srcinfo_mult(jpeg_decompress_struct *srcinfo,
//                                         bool normalized, int channels,
//                                         band_info *band1, band_info *band2,
//                                         band_info *band3,unsigned char *buffer){
//   unsigned int totalMcuNum = srcinfo->MCU_rows_in_scan * srcinfo->MCUs_per_row;
//   unsigned int totalBlock = srcinfo->blocks_in_MCU * totalMcuNum;
//     short *inputBuffer = (short*)malloc(totalBlock * sizeof(JBLOCK));
//   JBLOCK **mcu_buffer = (JBLOCK**)malloc(sizeof(JBLOCK*) * srcinfo->blocks_in_MCU);
//   JBLOCK *block_buffer = (JBLOCK*)inputBuffer;
//   pthread_t thread1, thread2,thread3,thread4;
//   struct thread_data t1_data = {0, flag1 , srcinfo, mcu_buffer, block_buffer,7,1};
//   struct thread_data t2_data = {flag1, flag2, srcinfo, mcu_buffer, block_buffer,6,2};
//   struct thread_data t3_data = {flag2, flag3, srcinfo, mcu_buffer, block_buffer,5,3};
//   struct thread_data t4_data = {flag3, (int)totalMcuNum, srcinfo, mcu_buffer, block_buffer,4,4};
//   srcinfo->unread_marker = 0;
//   srcinfo->unread_marker2 = 0;
//   srcinfo->unread_marker3 = 0;
//   srcinfo->unread_marker4 = 0;
//   srcinfo->now_b1_id=1;
//   srcinfo->now_b2_id=srcinfo->huff_scan.buffer_id[flag1]-1;//di er ge xian cheng kai shi de buffer id;
//   srcinfo->now_b3_id=srcinfo->huff_scan.buffer_id[flag2]-1;
//   srcinfo->now_b4_id=srcinfo->huff_scan.buffer_id[flag3]-1;
//   (*srcinfo->src->retrun_file_start) (srcinfo);
//   // 创建线程1
//   pthread_create(&thread1, NULL, process_mcus, &t1_data);
//   // 创建线程2
//   pthread_create(&thread2, NULL, process_mcus, &t2_data);
//   // 创建线程1
//   pthread_create(&thread3, NULL, process_mcus, &t3_data);
// //   // 创建线程2
//   pthread_create(&thread4, NULL, process_mcus, &t4_data);

//   // 等待两个线程完成
//   pthread_join(thread1, NULL);

//   pthread_join(thread2, NULL);

//   pthread_join(thread3, NULL);
//       //usleep(50000);
//   pthread_join(thread4, NULL);
//     band1->dct_h = srcinfo->comp_info[0].height_in_blocks;
//   band1->dct_w = srcinfo->comp_info[0].width_in_blocks;
//   band1->dct_b = DCTSIZE2;
//   long nb_elements1 =
//       (long)(band1->dct_h) * (long)(band1->dct_w) * (long)(band1->dct_b);
//   band1->dct = new short[nb_elements1];// 分配内存空间
//       band2->dct_h = srcinfo->comp_info[1].height_in_blocks;
//   band2->dct_w = srcinfo->comp_info[1].width_in_blocks;
//   band2->dct_b = DCTSIZE2;
//   long nb_elements2 =
//       (long)(band2->dct_h) * (long)(band2->dct_w) * (long)(band2->dct_b);
//   band2->dct = new short[nb_elements2];// 分配内存空间
//       band3->dct_h = srcinfo->comp_info[2].height_in_blocks;
//   band3->dct_w = srcinfo->comp_info[2].width_in_blocks;
//   band3->dct_b = DCTSIZE2;
//   long nb_elements3 =
//       (long)(band3->dct_h) * (long)(band3->dct_w) * (long)(band3->dct_b);
//   band3->dct = new short[nb_elements3];// 分配内存空间
//   auto start = std::chrono::high_resolution_clock::now();
//   struct result_data r1 = {0,srcinfo,inputBuffer,7,normalized,totalBlock,band1,0};
//   struct result_data r3 = {0,srcinfo,inputBuffer,6,normalized,totalBlock,band1,1};
//   struct result_data2 r2 = {srcinfo,inputBuffer,5,normalized,totalBlock,band2,band3,0};

//   pthread_create(&thread1, NULL, process_result, &r1);
//   pthread_create(&thread3, NULL, process_result, &r3);
//   pthread_create(&thread2, NULL, process_result2, &r2);

//   pthread_join(thread1, NULL);

//   pthread_join(thread2, NULL);
//   pthread_join(thread3, NULL);
//   if (buffer != nullptr) {
//     delete[] buffer;
//   }
// }
// void read_dct_coefficients_from_file_scan(char *filename,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
  
//   if ((infile = fopen(filename, "rb")) == nullptr) {
//     fprintf(stderr, "ERROR: can't open %s\n", filename);
//     return;
//   }

//   jpeg_error_mgr jerr;
//   srcinfo->err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_decompress(srcinfo);
//   jpeg_stdio_src(srcinfo, infile);
  
//   jpeg_read_header(srcinfo, TRUE);
//   jpeg_start_decompress(srcinfo);

//   if (!is_grayscale(srcinfo) && !is_h2_v2(srcinfo)) {// 如果图像不是灰度且不是 H2V2 格式，则进行转码
//     transcode(srcinfo, &buffer);
//   }
//   // 读取 JPEG 文件的 DCT 系数
//   jpeg_read_coefficients_scan(srcinfo);

// }
// void read_dct_coefficients_from_file_mult(char *filename, bool normalized,
//                                       int channels, band_info *band1,
//                                       band_info *band2, band_info *band3,FILE *infile,struct jpeg_decompress_struct *srcinfo,unsigned char *buffer){
//   read_dct_coefficients_from_srcinfo_mult(srcinfo,normalized,channels,band1,band2,band3,buffer);
//   jpeg_abort_decompress(srcinfo);
//   jpeg_destroy_decompress(srcinfo);
//   //fclose(infile);
// }



// void runInference(const std::vector<float>& input1_batch, 
//                               const std::vector<float>& input2_batch, 
//                               const std::vector<float>& input3_batch) {
//     std::vector<float> merged_input = mergeInputs(input2_batch, input3_batch);
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];
//     //int input_index3 = interpreter->inputs()[2];

//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);

//     std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.f, merged_input.data(), merged_input.size());
//     // 4. Run inference
//     interpreter->Invoke();
    
//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     // 假设 output_data_uint8 是指向量化输出数据的指针
//     const float* output_data= output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组
// }
// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//   cpu_set_t mask;
//   CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
//   CPU_SET(7, &mask);
//   if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
//    printf("Set CPU affinity failue, ERROR:%s\n", strerror(errno));
//    }
//     const char* model_path = argv[1];
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return 0;
//     }
//     // 创建解释器
//     tflite::ops::builtin::BuiltinOpResolver* resolver = nullptr;
//     resolver = new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
//     tflite::InterpreterBuilder builder(*model, *resolver);
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//         return 0;
//     }
// //创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }
//     interpreter->SetNumThreads(4);
//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return 0;
//     }
//     ctpl::thread_pool cpu1_thread_(1,7);
//     std::future<void> cpu1_future;

//     char* files[8] = {
//         "83.jpeg", "84.jpeg", "85.jpeg",
//         "86.jpeg", "87.jpeg", "88.jpeg",
//         "89.jpeg", "90.jpeg"
//     };
//     auto start_all = std::chrono::high_resolution_clock::now();
//     const int num_iterations = 1;
//       long long total_duration = 0;
//       long long total_duration_invoke = 0;
//       long long total_duration_all = 0;
//       int size = 1000*1000*64;

//     for (int i = 0; i < num_iterations; ++i) {

//         auto start_prepare_data = std::chrono::high_resolution_clock::now();
//         char* filename = files[i%8];
//         //char* filename = "83.jpeg";
//         short* band1_dct = new short[size];
//         int* band1_dct_h = new int(0);
//         int* band1_dct_w = new int(0);
//         int* band1_dct_b = new int(0);
//         short* band2_dct = new short[size];
//         int* band2_dct_h = new int(0);
//         int* band2_dct_w = new int(0);
//         int* band2_dct_b = new int(0);
//         short* band3_dct = new short[size];
//         int* band3_dct_h = new int(0);
//         int* band3_dct_w = new int(0);
//         int* band3_dct_b = new int(0);
//         bool normalized = true;  // Example value
//         int channels = 3;        // Example value
//         FILE* infile = nullptr;
//         band_info* band1 = new band_info();
//         band_info* band2 = new band_info();
//         band_info* band3 = new band_info();
//         jpeg_decompress_struct* srcinfo = new jpeg_decompress_struct();
//         unsigned char* buffer = nullptr;
//         //std::cout<<"end------------------"<<std::endl;
//         read_dct_coefficients_from_file_scan(filename,infile,srcinfo,buffer);
//         auto end_scan= std::chrono::high_resolution_clock::now();
//         //std::cout<<"end------------------0"<<std::endl;
//         read_dct_coefficients_from_file_mult(filename,normalized,channels,band1,band2,band3,infile,srcinfo,buffer);
//         unpack_band_info(band1, &band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
//         unpack_band_info(band2, &band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
//         unpack_band_info(band3, &band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
//         //std::cout<<"end------------------1"<<std::endl;
//         int targetWidth1 = 28;
//         int targetHeight1= 28;
//         int targetWidth2 = 14;
//         int targetHeight2 = 14;
//         short* resized_data1 = resizeShortData(band1_dct, *band1_dct_h, *band1_dct_w, *band1_dct_b, targetWidth1, targetHeight1);
//         short* resized_data2 = resizeShortData(band2_dct, *band2_dct_h, *band2_dct_w, *band2_dct_b, targetWidth2, targetHeight2);
//         short* resized_data3 = resizeShortData(band3_dct, *band3_dct_h, *band3_dct_w, *band3_dct_b, targetWidth2, targetHeight2);

//         std::vector<float> input1 = convertToUint8Vector(resized_data1, targetWidth1, targetHeight1, *band1_dct_b);
//         std::vector<float> input2 = convertToUint8Vector(resized_data2, targetWidth2, targetHeight2, *band2_dct_b);
//         std::vector<float> input3 = convertToUint8Vector(resized_data3, targetWidth2, targetHeight2, *band3_dct_b);

//         std::vector<float> input1_batch = addBatchDimension(input1,targetHeight1, targetWidth1, *band1_dct_b);
//         std::vector<float> input2_batch = addBatchDimension(input2,targetHeight2, targetWidth2, *band2_dct_b);
//         std::vector<float> input3_batch = addBatchDimension(input3,targetHeight2, targetWidth2, *band3_dct_b);
//         auto end_all_prepare = std::chrono::high_resolution_clock::now();
//         //runInference(input1_batch, input2_batch, input3_batch);
//         auto end_all = std::chrono::high_resolution_clock::now();
//         // delete[] band1_dct;
//         // delete[] band2_dct;
//         // delete[] band3_dct;
//         // 计算持续时间并转换为毫秒
//         // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_all_prepare - start_prepare_data);
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all_prepare - start_prepare_data);
//         auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - end_all_prepare);
//         auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_prepare_data);
//          total_duration += duration.count();
//          total_duration_invoke +=duration_invoke.count();
//          total_duration_all +=duration_all.count();
//          auto du_scan =std::chrono::duration_cast<std::chrono::milliseconds>(end_scan - start_prepare_data);
//           //std::cout<<"scan:"<<du_scan.count()<<std::endl;
//     }
//     //outFile.close();
//      auto end_all = std::chrono::high_resolution_clock::now();
//      auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
//     std::cout << "average 时间: " << total_duration_all /num_iterations<< " 毫秒" << std::endl;
//     std::cout<<"invoke:"<<total_duration_invoke/num_iterations<<std::endl;
//     std::cout<<"prepare:"<<total_duration/num_iterations<<std::endl;
//     std::cout << " end:"<<duration_all.count()<<std::endl;
    
//     return 0;
// }


// #include <thread>
// #include <sched.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <jpeglib.h>
// #include <future>
// #include <mutex>
// #include <condition_variable>
// #include <unordered_map>
// #include <atomic>
// #include <queue>
// #include <functional>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "ctpl_stl.h"
// #include "tensorflow/lite/delegates/gpu/delegate.h"
// #include <cstdlib>
// #include <cstring>
// #include <stdexcept>
// #include <stdio.h>
// #include <unistd.h>
// #include <sys/syscall.h>
// #include <pthread.h>
// #include "jpegint.h"
// struct band_info {
//   short *dct;
//   unsigned int dct_h;
//   unsigned int dct_w;
//   unsigned int dct_b;
// };

// std::unique_ptr<tflite::Interpreter> interpreter;
// std::vector<float> addBatchDimension(const std::vector<float>& imageData, int height, int width, int channels) {
//     std::vector<float> batchedImage(1 * height * width * channels);
//     std::memcpy(batchedImage.data(), imageData.data(), imageData.size());
//     return batchedImage;
// }
// //using namespace jpeg2dct::common;
// std::vector<float> convertToFloat32Vector(short* band_dct, int dct_h, int dct_w, int dct_b) {
//     std::vector<float> uint8_vector;
//     long nb_elements = (long)(dct_h) * (long)(dct_w) * (long)(dct_b);
//     uint8_vector.reserve(nb_elements);
//     for (long i = 0; i < nb_elements; ++i) {
//         uint8_vector.push_back(static_cast<float>(band_dct[i]));
//     }
//     return uint8_vector;
// }

// // Updated resizeImage function for NHWC format
// short* resizeShortData(const short* input_data, int original_height, int original_width, int channels,
//                        int new_height, int new_width) {
//     // 动态分配内存给 resized_data
//     short* resized_data = new short[new_height * new_width * channels];

//     // 计算缩放比例
//     float scale_h = static_cast<float>(original_height - 1) / (new_height - 1);
//     float scale_w = static_cast<float>(original_width - 1) / (new_width - 1);

//     // 对每个像素进行插值
//     for (int y = 0; y < new_height; ++y) {
//         for (int x = 0; x < new_width; ++x) {
//             float gy = y * scale_h;
//             float gx = x * scale_w;

//             int gxi = static_cast<int>(gx);
//             int gyi = static_cast<int>(gy);

//             for (int c = 0; c < channels; ++c) {
//                 float c00 = input_data[(gyi * original_width + gxi) * channels + c];
//                 float c10 = input_data[(gyi * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];
//                 float c01 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + gxi) * channels + c];
//                 float c11 = input_data[(std::min(gyi + 1, original_height - 1) * original_width + std::min(gxi + 1, original_width - 1)) * channels + c];

//                 float cx0 = c00 + (c10 - c00) * (gx - gxi);
//                 float cx1 = c01 + (c11 - c01) * (gx - gxi);
//                 float cxy = cx0 + (cx1 - cx0) * (gy - gyi);

//                 resized_data[(y * new_width + x) * channels + c] = static_cast<short>(std::round(cxy));
//             }
//         }
//     }

//     return resized_data;
// }

// const char* cifar10_labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

// const int height = 14;
// const int width = 14;
// const int depth1 = 64;  // 第一个输入的通道数
// const int depth2 = 64;  // 第二个输入的通道数
// // 合并函数，将两个 14x14x64 的输入合并为一个 14x14x128
// std::vector<float> mergeInputs(const std::vector<float>& input1, const std::vector<float>& input2) {
//     // 检查输入大小是否匹配
//     if (input1.size() != height * width * depth1 || input2.size() != height * width * depth2) {
//         std::cerr << "输入大小不匹配!" << std::endl;
//         return {};
//     }

//     // 创建合并后的 vector，大小为14x14x128
//     std::vector<float> merged_input(height * width * (depth1 + depth2));

//     // 逐元素合并
//     for (int i = 0; i < height * width; ++i) {
//         // 将 input1 的数据拷贝到 merged_input 中
//         for (int d1 = 0; d1 < depth1; ++d1) {
//             merged_input[i * (depth1 + depth2) + d1] = input1[i * depth1 + d1];
//         }
//         // 将 input2 的数据拷贝到 merged_input 中
//         for (int d2 = 0; d2 < depth2; ++d2) {
//             merged_input[i * (depth1 + depth2) + depth1 + d2] = input2[i * depth2 + d2];
//         }
//     }

//     return merged_input;
// }
// void unpack_band_info(band_info band, short **band_dct, int *band_dct_h,
//                       int *band_dct_w, int *band_dct_b) {
//   *band_dct = band.dct;
//   *band_dct_h = band.dct_h;
//   *band_dct_w = band.dct_w;
//   *band_dct_b = band.dct_b;
// }

// void dummy_dct_coefficients(band_info *band) {
//   band->dct_h = 0;
//   band->dct_w = 0;
//   band->dct_b = 0;
//   band->dct = new short[0];
// }

// void read_dct_coefficients(jpeg_decompress_struct *srcinfo,
//                            short *inputBuffer, int compNum,
//                            band_info *band, bool normalized,unsigned int totalBlock) {
//   if (compNum >= srcinfo->num_components) {// 检查通道索引是否合法
//     // make an empty component which would be half size of chroma // 创建一个空的组件，大小为色度的一半
//     band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2; //一个block高
//     band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;//一个block宽
//     band->dct_b = DCTSIZE2;//  可以调节输出通道数 每个block大小
//     long nb_elements =
//         (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);//有多上个元素
//     band->dct = new short[nb_elements];// 分配内存空间并初始化为 0
//     std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
//     return;
//   }

//   // prepare memory space dimensions // 准备内存空间的维度
//   band->dct_h = srcinfo->comp_info[compNum].height_in_blocks;
//   band->dct_w = srcinfo->comp_info[compNum].width_in_blocks;
//   band->dct_b = DCTSIZE2;
//   long nb_elements =
//       (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);
//   band->dct = new short[nb_elements];// 分配内存空间

//   int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
//   short unscale = 1;

//   short *current_dct_coeff = band->dct;
//   if(compNum==0){
//       for(int i =0; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//                 if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                   unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//                 }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+5;
//     }
//     for(int i =2; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i++;
//       short* src2 = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+5;
//     }
//   }
//   else{
//       for(int i =compNum+3; i<totalBlock;){
//       short* src = inputBuffer + 64 * i;
//       for (unsigned int j =0 ;j<64;j++){
//               if (normalized) {// 如果需要标准化，则使用量化表进行反量化
//                 unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[j];
//               }
//               *current_dct_coeff=src[j]*unscale;
//               current_dct_coeff++;
//         }
//       i=i+6;
//     }
//   }

// }

// void add_error_handler(jpeg_error_mgr *jerr) {
//   jerr->error_exit = [](j_common_ptr cinfo) {
//     char pszErr[1024];
//     (cinfo->err->format_message)(cinfo, pszErr);
//     fprintf(stderr, "Error: %s\n", pszErr);
//     exit(EXIT_FAILURE);
//   };
// }


// void transcode(jpeg_decompress_struct *srcinfo, unsigned char **outbuffer) {
//   static bool warning_emitted = false;
//   if (!warning_emitted) {// 发出警告，提示遇到非标准的 JPEG 图像，需要转码
//     fprintf(stderr, "WARNING: Non-standard JPEG image encountered, transcoding "
//                     "to H2_V2 which may negatively impact performance. This is "
//                     "the only time this warning will be shown.\n");
//     warning_emitted = true;
//   }

//   // start decompress
//   (void)jpeg_start_decompress(srcinfo);

//   // create the compression structure
//   jpeg_compress_struct dstinfo;
//   jpeg_error_mgr jerr;
//   dstinfo.err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_compress(&dstinfo);

//   unsigned long outlen = 0;
//   jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

//   dstinfo.image_width = srcinfo->image_width;
//   dstinfo.image_height = srcinfo->image_height;
//   dstinfo.input_components = srcinfo->output_components;
//   dstinfo.in_color_space = srcinfo->out_color_space;

//   jpeg_set_defaults(&dstinfo);
//   jpeg_set_quality(&dstinfo, 100, TRUE);
//   jpeg_start_compress(&dstinfo, TRUE);

//   // transcode// 转码  读出来立马写回去
//   unsigned char *line_buffer =
//       new unsigned char[srcinfo->output_width * srcinfo->output_components];
//   while (srcinfo->output_scanline < srcinfo->output_height) {
//     jpeg_read_scanlines(srcinfo, &line_buffer, 1);
//     (void)jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
//   }
//   delete[] line_buffer;
//   jpeg_finish_compress(&dstinfo);
//   jpeg_destroy_compress(&dstinfo);

//   // re-create decompress
//   jpeg_destroy_decompress(srcinfo);
//   jpeg_create_decompress(srcinfo);
//   jpeg_mem_src(srcinfo, *outbuffer, outlen);
//   (void)jpeg_read_header(srcinfo, TRUE);
// }

// bool is_grayscale(jpeg_decompress_struct *srcinfo) {
//   return srcinfo->num_components == 1;
// }

// bool is_h2_v2(jpeg_decompress_struct *srcinfo) {
//   return (srcinfo->comp_info[0].h_samp_factor == 2) &&
//          (srcinfo->comp_info[0].v_samp_factor == 2) &&
//          (srcinfo->comp_info[1].h_samp_factor == 1) &&
//          (srcinfo->comp_info[1].v_samp_factor == 1) &&
//          (srcinfo->comp_info[2].h_samp_factor == 1) &&
//          (srcinfo->comp_info[2].v_samp_factor == 1);
// }

// void read_dct_coefficients_from_srcinfo(jpeg_decompress_struct *srcinfo,
//                                         bool normalized, int channels,
//                                        band_info *band1, band_info *band2,
//                                         band_info *band3) {
//   unsigned char *buffer = nullptr;
//   unsigned int totalMcuNum = srcinfo->MCU_rows_in_scan * srcinfo->MCUs_per_row;
//   unsigned int totalBlock = srcinfo->blocks_in_MCU * totalMcuNum;
//   short *inputBuffer = (short*)malloc(totalBlock * sizeof(JBLOCK));
//     JBLOCK **mcu_buffer = (JBLOCK**)malloc(sizeof(JBLOCK*) * srcinfo->blocks_in_MCU);
//     JBLOCK *block_buffer = (JBLOCK*)inputBuffer;
//     for (int i = 0; i < totalMcuNum; ++i) {
//         for (int j = 0; j < srcinfo->blocks_in_MCU; ++j) {
//             mcu_buffer[j] = block_buffer + srcinfo->blocks_in_MCU * i + j;
//         }
//         (srcinfo->entropy->decode_mcu)(srcinfo, mcu_buffer);
//     }
//     free(mcu_buffer);
//   read_dct_coefficients(srcinfo, inputBuffer, 0, band1, normalized,totalBlock);// 读取第一个通道的 DCT 系数
//   if (channels == 3) {// 如果通道数为 3，则继续读取第二个和第三个通道的 DCT 系数，否则将其设置为虚拟的 DCT 系数
//     read_dct_coefficients(srcinfo, inputBuffer, 1, band2, normalized,totalBlock);
//     read_dct_coefficients(srcinfo, inputBuffer, 2, band3, normalized,totalBlock);
//   } else {
//     dummy_dct_coefficients(band2);
//     dummy_dct_coefficients(band3);
//   }
//   if (buffer != nullptr) {
//     delete[] buffer;
//   }
// }

// void read_dct_coefficients_from_file_(char *filename, bool normalized,
//                                       int channels, band_info *band1,
//                                       band_info *band2, band_info *band3) {
//   FILE *infile;
//   if ((infile = fopen(filename, "rb")) == nullptr) {
//     fprintf(stderr, "ERROR: can't open %s\n", filename);
//     return;
//   }

//   jpeg_decompress_struct srcinfo;
//   jpeg_error_mgr jerr;
//   srcinfo.err = jpeg_std_error(&jerr);
//   add_error_handler(&jerr);
//   jpeg_create_decompress(&srcinfo);
//   jpeg_stdio_src(&srcinfo, infile);

//   jpeg_read_header(&srcinfo, TRUE);
//   jpeg_start_decompress(&srcinfo);
//   read_dct_coefficients_from_srcinfo(&srcinfo, normalized, channels, band1,
//                                      band2, band3);

//   jpeg_destroy_decompress(&srcinfo);
//   fclose(infile);
// }

// void read_dct_coefficients_from_file_1(
//     char *filename, bool normalized, int channels, short **band1_dct,
//     int *band1_dct_h, int *band1_dct_w, int *band1_dct_b, short **band2_dct,
//     int *band2_dct_h, int *band2_dct_w, int *band2_dct_b, short **band3_dct,
//     int *band3_dct_h, int *band3_dct_w, int *band3_dct_b) {
//   band_info band1, band2, band3;
//   read_dct_coefficients_from_file_(filename, normalized, channels, &band1,
//                                    &band2, &band3);
//   unpack_band_info(band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
//   unpack_band_info(band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
//   unpack_band_info(band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
// }

// void runInference(const std::vector<float>& input1_batch, 
//                               const std::vector<float>& input2_batch, 
//                               const std::vector<float>& input3_batch) {
//     std::vector<float> merged_input = mergeInputs(input2_batch, input3_batch);
//     int input_index1 = interpreter->inputs()[0];
//     int input_index2 = interpreter->inputs()[1];

//     TfLiteTensor* tensor1 = interpreter->tensor(input_index1);
//     TfLiteTensor* tensor2 = interpreter->tensor(input_index2);
//     std::memcpy(tensor1->data.f, input1_batch.data(), input1_batch.size());
//     std::memcpy(tensor2->data.f, merged_input.data(), merged_input.size());
//     // 4. Run inference
//     interpreter->Invoke();
    
//     // 5. Process output tensors
//     // Assuming output tensor is at index 0
//     int output_tensor_idx = interpreter->outputs()[0];
//     TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
//     const float* output_data= output_tensor->data.f;
//     int output_size = output_tensor->dims->data[1]; // 假设输出是一个1D数组
// }
// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_filename>" << std::endl;
//         return 1;
//     }
//     const char* model_path = argv[1];
//     std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
//     if (!model) {
//         std::cerr << "Failed to load model: " << model_path << std::endl;
//         return 0;
//     }
//     // 创建解释器
//     //tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::ops::builtin::BuiltinOpResolver* resolver = nullptr;
//     resolver = new tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
//     tflite::InterpreterBuilder builder(*model, *resolver);
//     builder(&interpreter);
//     if (!interpreter) {
//         std::cerr << "Failed to build interpreter." << std::endl;
//         return 0;
//     }
// //创建GPU代理
//     TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
//     // 创建并绑定代理
//     auto* delegate = TfLiteGpuDelegateV2Create(&options);
//     if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
//         std::cerr << "Failed to apply GPU delegate." << std::endl;
//         TfLiteGpuDelegateV2Delete(delegate);
//         return -1;
//     }
//     interpreter->SetNumThreads(4);
//     // 分配张量
//     if (interpreter->AllocateTensors() != kTfLiteOk) {
//         std::cerr << "Failed to allocate tensors." << std::endl;
//         return 0;
//     }
//     ctpl::thread_pool cpu1_thread_(1,7);
//     std::future<void> cpu1_future;

//     char* files[8] = {
//         "83.jpeg", "84.jpeg", "85.jpeg",
//         "86.jpeg", "87.jpeg", "88.jpeg",
//         "89.jpeg", "90.jpeg"
//     };
//     auto start_all = std::chrono::high_resolution_clock::now();
//     const int num_iterations = 50;
//       long long total_duration = 0;
//       long long total_duration_invoke = 0;
//       long long total_duration_all = 0;
//     for (int i = 0; i < num_iterations; ++i) {

//         auto start_prepare_data = std::chrono::high_resolution_clock::now();
//         char* filename = files[i%8];
//         //char* filename = "83.jpeg";
//         short *band1_dct, *band2_dct, *band3_dct;
//         int band1_dct_h, band1_dct_w, band1_dct_b;
//         int band2_dct_h, band2_dct_w, band2_dct_b;
//         int band3_dct_h, band3_dct_w, band3_dct_b;
        
//         // 假设 read_dct_coefficients_from_file 已经实现并且返回正确的数据
//       read_dct_coefficients_from_file_1(filename, true, 3,
//                                       &band1_dct, &band1_dct_h, &band1_dct_w, &band1_dct_b,
//                                       &band2_dct, &band2_dct_h, &band2_dct_w, &band2_dct_b,
//                                       &band3_dct, &band3_dct_h, &band3_dct_w, &band3_dct_b);
        
//         int targetWidth1 = 28;
//         int targetHeight1= 28;
//         int targetWidth2 = 14;
//         int targetHeight2 = 14;
//         short* resized_data1 = resizeShortData(band1_dct, band1_dct_h, band1_dct_w, band1_dct_b, targetWidth1, targetHeight1);
//         short* resized_data2 = resizeShortData(band2_dct, band2_dct_h, band2_dct_w, band2_dct_b, targetWidth2, targetHeight2);
//         short* resized_data3 = resizeShortData(band3_dct, band3_dct_h, band3_dct_w, band3_dct_b, targetWidth2, targetHeight2);

//         std::vector<float> input1 = convertToFloat32Vector(band1_dct, targetWidth1, targetHeight1, band1_dct_b);
//         std::vector<float> input2 = convertToFloat32Vector(band2_dct, targetWidth2, targetHeight2, band2_dct_b);
//         std::vector<float> input3 = convertToFloat32Vector(band3_dct, targetWidth2, targetHeight2, band3_dct_b);

//         std::vector<float> input1_batch = addBatchDimension(input1,targetHeight1, targetWidth1, band1_dct_b);
//         std::vector<float> input2_batch = addBatchDimension(input2,targetHeight2, targetWidth2, band2_dct_b);
//         std::vector<float> input3_batch = addBatchDimension(input3,targetHeight2, targetWidth2, band3_dct_b);
//         auto end_all_prepare = std::chrono::high_resolution_clock::now();
//         // if (cpu1_future.valid()) {
//         //     cpu1_future.wait();  
//         // }

//         // cpu1_future = cpu1_thread_.push([&input1_batch,&input2_batch, &input3_batch](int) { 
//         //     runInference(input1_batch, input2_batch, input3_batch); // CPU1
//         // });
//         // if(i==num_iterations-1){
//         // cpu1_future.wait();
//         // }
//         runInference(input1_batch, input2_batch, input3_batch);
//         auto end_all = std::chrono::high_resolution_clock::now();
//         delete[] band1_dct;
//         delete[] band2_dct;
//         delete[] band3_dct;
//         // 计算持续时间并转换为毫秒
//         // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_all_prepare - start_prepare_data);
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_all_prepare - start_prepare_data);
//         auto duration_invoke = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - end_all_prepare);
//         auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_prepare_data);
//          total_duration += duration.count();
//          total_duration_invoke +=duration_invoke.count();
//          total_duration_all +=duration_all.count();
//     }
//     //outFile.close();
//      auto end_all = std::chrono::high_resolution_clock::now();
//      auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all);
//     std::cout << "average 时间: " << total_duration_all /num_iterations<< " 毫秒" << std::endl;
//     std::cout << "all时间: " << total_duration_all << " 毫秒" << std::endl;
//     std::cout << " per:"<<total_duration/num_iterations<<std::endl;
//     std::cout<<"invoke:"<<total_duration_invoke/num_iterations<<std::endl;
    
//     return 0;
// }

