#CFLAGS=-I/opt/opencv/include -I/opt/dlib/include -I/opt/libtorch/x64/include -I/opt/libtorch/x64/include/torch/csrc/api/include -DUSE_EXTERNAL_MZCRC -DUSE_PTHREADPOOL -DNDEBUG -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -O3 -g -fPIC -Wall -pthread -fmessage-length=0
CXXFLAGS=-I/opt/opencv/include -I/opt/dlib/include -O3 -g -fPIC -pthread -fmessage-length=0 -std=c++14
LDFLAGS=-L/opt/opencv/lib -L/opt/dlib/lib -L/usr/local/cuda/lib64 -Wl,-rpath,/opt/opencv/lib:/usr/local/cuda/lib64 -ldlib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_objdetect -lopencv_ml -lopenblas -lcudart_static -lX11 -lpng -ljpeg -ldl -lrt -lpthread
OBJS=hog_pca_svm_train.o
TARGET=hog_pca_svm_train

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
