/*
 * benchmark_conv2d.cu  --  CPU vs CUDA Conv2D benchmarks
 * Shapes match SqueezeNet 1.1 layers.  Reports kernel-only and E2E separately.
 */
#include "forgert/operator/conv2d.h"
#include "forgert/tensor/tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <functional>
using namespace forgert;

#define CUDA_CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    std::cerr<<"CUDA "<<cudaGetErrorString(e)<<"\n"; std::exit(1);} } while(0)

static double timeCPU(std::function<void()> fn, int warmup=5, int iters=20) {
    for(int i=0;i<warmup;++i) fn();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) fn();
    auto t1=std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
}
template<typename Fn>
static double timeGPU(Fn fn, int warmup=10, int iters=50) {
    for(int i=0;i<warmup;++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t ev_s, ev_e; CUDA_CHECK(cudaEventCreate(&ev_s)); CUDA_CHECK(cudaEventCreate(&ev_e));
    CUDA_CHECK(cudaEventRecord(ev_s));
    for(int i=0;i<iters;++i) fn();
    CUDA_CHECK(cudaEventRecord(ev_e)); CUDA_CHECK(cudaEventSynchronize(ev_e));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,ev_s,ev_e));
    CUDA_CHECK(cudaEventDestroy(ev_s)); CUDA_CHECK(cudaEventDestroy(ev_e));
    return ms/iters;
}

struct Row { std::string label; int N,Ci,H,W,Co,kH,kW,sH,pH;
             double cpu_ms, kern_ms, e2e_ms; };

static Row bench(const char* label,
                 int N,int Ci,int H,int W,int Co,int kH,int sH,int pH) {
    Conv2DParams p; p.kH=kH; p.kW=kH; p.stride_h=sH; p.stride_w=sH;
                   p.pad_h=pH; p.pad_w=pH;
    Conv2DOp op(p);
    int Ho=(H+2*pH-kH)/sH+1, Wo=(W+2*pH-kH)/sH+1;

    TensorShape si({(size_t)N,(size_t)Ci,(size_t)H,(size_t)W});
    TensorShape sw({(size_t)Co,(size_t)Ci,(size_t)kH,(size_t)kH});
    TensorShape sb({(size_t)Co});
    TensorShape so({(size_t)N,(size_t)Co,(size_t)Ho,(size_t)Wo});

    Tensor hi(si,DataType::Float32,Device::CPU); hi.fill(0.1f);
    Tensor hw(sw,DataType::Float32,Device::CPU); hw.fill(0.01f);
    Tensor hb(sb,DataType::Float32,Device::CPU); hb.fill(0.0f);
    Tensor ho(so,DataType::Float32,Device::CPU);

    Tensor di(si,DataType::Float32,Device::CUDA); hi.copyTo(di);
    Tensor dw(sw,DataType::Float32,Device::CUDA); hw.copyTo(dw);
    Tensor db(sb,DataType::Float32,Device::CUDA); hb.copyTo(db);
    Tensor dout(so,DataType::Float32,Device::CUDA);

    double cpu_ms = timeCPU([&](){
        op.execute(Backend::CPU,{&hi,&hw,&hb},{&ho});
    });
    double kern_ms = timeGPU([&](){
        op.execute(Backend::CUDA,{&di,&dw,&db},{&dout});
    });
    double e2e_ms = timeGPU([&](){
        hi.copyTo(di); hw.copyTo(dw);
        op.execute(Backend::CUDA,{&di,&dw,&db},{&dout});
        dout.copyTo(ho);
    });
    return {label,N,Ci,H,W,Co,kH,kH,sH,pH,cpu_ms,kern_ms,e2e_ms};
}

int main() {
    {Tensor d(TensorShape({1}),DataType::Float32,Device::CUDA); CUDA_CHECK(cudaDeviceSynchronize());}
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"Conv2D Benchmark -- "<<prop.name<<"\n";
    std::cout<<"Warmup:10 iters, Measure:50 iters, CPU:20 iters\n\n";

    std::cout<<std::left
             <<std::setw(28)<<"Layer"
             <<std::setw(10)<<"Elems(K)"
             <<std::setw(12)<<"CPU(ms)"
             <<std::setw(14)<<"Kernel(ms)"
             <<std::setw(12)<<"E2E(ms)"
             <<"CPU/Kernel\n"
             <<std::string(86,'-')<<"\n";

    std::vector<Row> rows = {
        bench("conv0: 3x3 s2 [3,224,224]->64",   1,3,224,224,64,3,2,0),
        bench("conv1x1: [64,55,55]->16",           1,64,55,55,16,1,1,0),
        bench("conv1x1: [16,55,55]->64",           1,16,55,55,64,1,1,0),
        bench("conv3x3: [16,55,55]->64",           1,16,55,55,64,3,1,1),
        bench("conv1x1: [128,27,27]->32",          1,128,27,27,32,1,1,0),
        bench("conv3x3: [32,27,27]->128",          1,32,27,27,128,3,1,1),
        bench("conv1x1: [256,13,13]->48",          1,256,13,13,48,1,1,0),
        bench("conv3x3: [48,13,13]->192",          1,48,13,13,192,3,1,1),
        bench("conv1x1: [512,13,13]->1000",        1,512,13,13,1000,1,1,0),
    };

    for(const auto& r : rows) {
        size_t elems = (size_t)r.N*r.Co;
        int Ho=(r.H+2*r.pH-r.kH)/r.sH+1;
        elems *= Ho * ((r.W+2*r.pH-r.kH)/r.sH+1);
        double ratio = r.kern_ms>0 ? r.cpu_ms/r.kern_ms : 0;
        std::cout<<std::setw(28)<<r.label
                 <<std::setw(10)<<std::fixed<<std::setprecision(1)<<elems/1000.0
                 <<std::setw(12)<<std::setprecision(3)<<r.cpu_ms
                 <<std::setw(14)<<std::setprecision(4)<<r.kern_ms
                 <<std::setw(12)<<std::setprecision(3)<<r.e2e_ms
                 <<std::setprecision(1)<<ratio<<"x\n";
    }

    std::cout<<"\nns/elem (CPU): based on cpu_ms / output_elems\n";
    for(const auto& r : rows) {
        int Ho=(r.H+2*r.pH-r.kH)/r.sH+1;
        size_t elems=(size_t)r.N*r.Co*Ho*((r.W+2*r.pH-r.kH)/r.sH+1);
        double ns_per_elem = r.cpu_ms*1e6 / elems;
        std::cout<<"  "<<std::setw(38)<<r.label
                 <<std::setprecision(1)<<ns_per_elem<<" ns/elem\n";
    }
    std::cout<<"\nDone.\n";
}