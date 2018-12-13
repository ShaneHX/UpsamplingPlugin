#include <iostream>
#include "UpsmapleKernel.h"
#include "UpsamplePlugin.h"

#include<cassert>
#include <cstring>

using namespace nvinfer1;

// Upsample plugin specific constants
namespace {
    static const char* UPSAMPLE_PLUGIN_VERSION{"1"};
    static const char* UPSAMPLE_PLUGIN_NAME{"UpsamplePlugin"};
}

// Static class fields initialization
PluginFieldCollection UpsamplePluginCreator::mFC{};
std::vector<PluginField> UpsamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(UpsamplePluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

UpsamplePlugin::UpsamplePlugin(const std::string name, int scale_factor, bool align_corners)
    : mLayerName(name)
    , mAlignCorners(align_corners)
    , mScaleFactor(scale_factor)
{
    // printf("UpsamplePlugin::UpsamplePlugin1\n");
    mInputShape.c() = -1;
    mInputShape.h() = -1;
    mInputShape.w() = -1;
    mInputVolume = 0;
}

UpsamplePlugin::UpsamplePlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    //printf("UpsamplePlugin::UpsamplePlugin2\n");
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mScaleFactor = readFromBuffer<int>(d);
    mAlignCorners = readFromBuffer<bool>(d);

    mInputVolume = readFromBuffer<size_t>(d);
    mInputShape.c() = readFromBuffer<int>(d);
    mInputShape.h() = readFromBuffer<int>(d);
    mInputShape.w() = readFromBuffer<int>(d);

    // writeToBuffer(d, mInputVolume);
    // writeToBuffer(d, mInputShape.c());
    // writeToBuffer(d, mInputShape.h());
    // writeToBuffer(d, mInputShape.w());


    // mInputShape.c() = -1;
    // mInputShape.h() = -1;
    // mInputShape.w() = -1;
    // mInputVolume = 0;
    //printf("length: %d\n", int(length));
    assert(d == (a + length));

}

const char* UpsamplePlugin::getPluginType() const
{
    //printf("UpsamplePlugin::getPluginType\n");
    return UPSAMPLE_PLUGIN_NAME;
}

const char* UpsamplePlugin::getPluginVersion() const
{
    //printf("UpsamplePlugin::getPluginVersion\n");
    return UPSAMPLE_PLUGIN_VERSION;
}

int UpsamplePlugin::getNbOutputs() const
{
    //printf("UpsamplePlugin::getNbOutputs\n");
    return 1;
}

Dims UpsamplePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    //printf("UpsamplePlugin::getOutputDimensions\n");
    assert(index == 0);
    assert(nbInputDims == 1);
    assert(inputs[0].nbDims == 4); 

    return nvinfer1::DimsNCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]*mScaleFactor, inputs[0].d[3]*mScaleFactor};
}

int UpsamplePlugin::initialize()
{
    //printf("UpsamplePlugin::initialize\n");
    return 0;
}


int UpsamplePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    //printf("UpsamplePlugin::enqueue\n");
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = UpsampleInference(stream, mInputVolume, 
                                batchSize, mInputShape.c(), mInputShape.h(), mInputShape.w(),
                                mScaleFactor, mAlignCorners,
                                inputs[0], output);

    return status;
}

size_t UpsamplePlugin::getSerializationSize() const
{
    //printf("UpsamplePlugin::getSerializationSize\n");
    return sizeof(mScaleFactor) + sizeof(mAlignCorners) + 
            sizeof(mInputVolume) + sizeof(mInputShape.c()) + 
            sizeof(mInputShape.h()) + sizeof(mInputShape.w());
}


void UpsamplePlugin::serialize(void* buffer) const 
{
    //printf("UpsamplePlugin::serialize\n");
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mScaleFactor);
    writeToBuffer(d, mAlignCorners);
    writeToBuffer(d, mInputVolume);
    writeToBuffer(d, mInputShape.c());
    writeToBuffer(d, mInputShape.h());
    writeToBuffer(d, mInputShape.w());
    
    //printf("------getSerializationSize: %d\n",int(getSerializationSize()));

    assert(d == a + getSerializationSize());
}

void UpsamplePlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    //printf("-----UpsamplePlugin::configureWithFormat\n");
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);
    assert(inputs[0].nbDims == 4);
    // Fetch volume for future enqueue() operations
    size_t volume = inputs[0].d[2]*mScaleFactor * inputs[0].d[3]*mScaleFactor;
    mInputVolume = volume;
    mInputShape.c() = inputs[0].d[1];
    mInputShape.h() = inputs[0].d[2];
    mInputShape.w() = inputs[0].d[3];
}

bool UpsamplePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    //printf("UpsamplePlugin::supportsFormat\n");
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void UpsamplePlugin::terminate() {}

void UpsamplePlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* UpsamplePlugin::clone() const
{
    return new UpsamplePlugin(mLayerName, mScaleFactor, mAlignCorners);
}

void UpsamplePlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* UpsamplePlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

UpsamplePluginCreator::UpsamplePluginCreator()
{
    //printf("UpsamplePluginCreator::UpsamplePluginCreator\n");
    // Describe UpsamplePlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("scaleFactor", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("alignCorners", nullptr, PluginFieldType::kINT8, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
const char* UpsamplePluginCreator::getPluginName() const
{
    //printf("UpsamplePluginCreator::getPluginName\n");
    return UPSAMPLE_PLUGIN_NAME;
}

const char* UpsamplePluginCreator::getPluginVersion() const
{
    //printf("UpsamplePluginCreator::getPluginVersion\n");
    return UPSAMPLE_PLUGIN_VERSION;
}

const PluginFieldCollection* UpsamplePluginCreator::getFieldNames()
{
    //printf("UpsamplePluginCreator::getFieldNames\n");
    return &mFC;
}

IPluginV2* UpsamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    //printf("UpsamplePluginCreator::createPlugin\n");
    int scaleFactor = 0;
    bool alignCorners = false;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++){
    
        if (strcmp(fields[i].name, "scaleFactor") == 0) {
            assert(fields[i].type == PluginFieldType::kINT8);
            scaleFactor = *(static_cast<const int8_t*>(fields[i].data));
        } 
        else if (strcmp(fields[i].name, "alignCorners") == 0) {
            assert(fields[i].type == PluginFieldType::kINT8);
            alignCorners = bool(*(static_cast<const int8_t*>(fields[i].data)));

        }
    }
    return new UpsamplePlugin(name, scaleFactor, alignCorners);
}

IPluginV2* UpsamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //printf("UpsamplePluginCreator::deserializePlugin\n");
    // This object will be deleted when the network is destroyed, which will
    // call UpsamplePlugin::destroy()
    return new UpsamplePlugin(name, serialData, serialLength);
}

void UpsamplePluginCreator::setPluginNamespace(const char* libNamespace) 
{
    //printf("UpsamplePluginCreator::setPluginNamespace\n");
    mNamespace = libNamespace;
}

const char* UpsamplePluginCreator::getPluginNamespace() const
{
    //printf("UpsamplePluginCreator::getPluginNamespace\n");
    return mNamespace.c_str();
}
