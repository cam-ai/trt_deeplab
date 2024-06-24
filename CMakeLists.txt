cmake_minimum_required(VERSION 3.11)
project(trt)
set(PROJECT_NAME trt)
set(CMAKE_CXX_STANDARD 14)
# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIR}/include) 
link_directories(${OpenCV_DIR}/lib/)  

set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio)


# CUDA
set(CUDA_PATH /usr/local/cuda-11.1)
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA include at ${CUDA_LIBRARIES}")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_PATH}/lib64)
link_directories(${CUDA_PATH}/lib64/stubs)

#TensorRT
set(TENSORRT_ROOT /media/myue/AHS/001_work/TensorRT-8.6.1.6/) ####                   
message(STATUS "Find tensorrt root  at ${TENSORRT_ROOT}")
include_directories(${TENSORRT_ROOT}/include)  
link_directories(${TENSORRT_ROOT}/lib/)


add_executable(${PROJECT_NAME} main.cpp deeplab_trt_model.cpp)

################target 
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ) 
# target_link_libraries(${PROJECT_NAME} ${CUDA_PATH}/lib64 ) 
# target_link_libraries(${PROJECT_NAME} ${CUDA_PATH}/lib64/stubs ) 

target_link_libraries(${PROJECT_NAME} ${TENSORRT_ROOT}/lib/libnvinfer.so)
target_link_libraries(${PROJECT_NAME} ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so)
target_link_libraries(${PROJECT_NAME} ${TENSORRT_ROOT}/lib/libnvparsers.so)
target_link_libraries(${PROJECT_NAME} ${TENSORRT_ROOT}/lib/libnvonnxparser.so)
target_link_libraries(${PROJECT_NAME} ${TENSORRT_ROOT}/lib/libnvonnxparser.so)

# set(NV_LIBS "")
set(NV_LIBS nvinfer cudart nvinfer_plugin nvparsers nvonnxparser cublas cublasLt cudnn)
target_link_libraries(${PROJECT_NAME} ${NV_LIBS} )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)