#include <assert.h>

#include <math.h>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

const int BATCH_SIZE = 16;

/**
 * Individual residual block
 *
 * @param convSourceDims Size of the input tensor, 
 * eg. (batch, channels, width, height)
 * @param convDestDims Size of the output tensor
 * @param downSample Whether the residual block should halve the
 * tensor size, true by default
 */
void resblock(
    engine::kind engineKind
) {
    // TODO WIP, these should be function params
    int convFilters = 64;
    memory::dims convSourceDims = {BATCH_SIZE, 3, 224, 224};
    memory::dims convDestDims {BATCH_SIZE, convFilters, 224, 224};
    bool downSample = false;

    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = engine(engineKind, 0);

    stream s(eng);

    // network primitives and their execute args
    std::vector<primitive> netForward, netBackward;
    std::vector<std::unordered_map<int, memory>> 
        netForwardArgs, netBackwardArgs;

    // tensors of data for the input and output
    // this should propably be passed to the method
    std::vector<float> sourceTensor(
        convSourceDims.at(0) * convSourceDims.at(1) *
        convSourceDims.at(2) * convSourceDims.at(3));

    std::vector<float> destTensor(
        convDestDims.at(0) * convDestDims.at(1) *
        convDestDims.at(2) * convDestDims.at(3));


    // -----------------------------
    //     1st convolution layer
    // -----------------------------

    // TODO add skip connection
    // TODO check if downsampling works

    // 3x3 convolution weights and biases initialized to sin(i) (nonzero)
    memory::dims conv1WeightsDims = {
        convFilters, convSourceDims.at(1), 3, 3};
    memory::dims conv1BiasDims = {convFilters};
    memory::dims conv1Strides = {downSample ? 2 : 1, downSample ? 2 : 1};
    memory::dims conv1Padding = {downSample ? 0 : 1, downSample ? 0 : 1};
    
    std::vector<float> conv1Weights(product(conv1WeightsDims));
    for (size_t i = 0; i < conv1Weights.size(); ++i) {
        conv1Weights[i] = sinf(float(i));
    }
    std::vector<float> conv1Bias(product(conv1BiasDims));
    for (size_t i = 0; i < conv1Bias.size(); ++i) {
        conv1Bias[i] = sinf(float(i));
    }

    // memory for tensor data
    auto conv1UserSourceMemory = 
        memory({{convSourceDims}, dt::f32, tag::nchw}, eng);
    auto conv1UserWeightsMemory =
        memory({{conv1WeightsDims}, dt::f32, tag::oihw}, eng);
    auto conv1BiasMemory =
        memory({{conv1BiasDims}, dt::f32, tag::x}, eng);
    
    write_to_dnnl_memory(sourceTensor.data(), conv1UserSourceMemory);
    write_to_dnnl_memory((void *) conv1Weights.data(), conv1UserWeightsMemory);
    write_to_dnnl_memory(conv1Bias.data(), conv1BiasMemory);

    // memory descriptors for convolution data
    auto conv1SourceMD = memory::desc({convSourceDims}, dt::f32, tag::any);
    auto conv1DestMD = memory::desc({convDestDims}, dt::f32, tag::any);
    auto conv1WeightsMD = memory::desc({conv1WeightsDims}, dt::f32, tag::any);
    auto conv1BiasMD = memory::desc({conv1BiasDims}, dt::f32, tag::any);

    // convolution primitive descriptor
    auto conv1PrimitiveDesc = convolution_forward::primitive_desc(
        eng, prop_kind::forward, algorithm::convolution_direct,
        conv1SourceMD, conv1WeightsMD, conv1BiasMD,
        conv1DestMD, conv1Strides, conv1Padding, conv1Padding
    );

    // reorder primitives between input and conv src if needed (??)
    // TODO check what that does 
    auto conv1SourceMemory = conv1UserSourceMemory;
    if (conv1PrimitiveDesc.src_desc() != conv1UserSourceMemory.get_desc()) {
        conv1UserSourceMemory = memory(conv1PrimitiveDesc.src_desc(), eng);
        netForward.push_back(reorder(conv1UserSourceMemory, conv1SourceMemory));
        netForwardArgs.push_back({{DNNL_ARG_FROM, conv1UserSourceMemory}, {DNNL_ARG_TO, conv1SourceMemory}});
    }

    auto conv1WeightsMemory = conv1UserWeightsMemory;
    if (conv1PrimitiveDesc.weights_desc() != conv1UserWeightsMemory.get_desc()) {
        conv1WeightsMemory = memory(conv1PrimitiveDesc.weights_desc(), eng);
        netForward.push_back(reorder(conv1UserWeightsMemory, conv1WeightsMemory));
        netForwardArgs.push_back({{DNNL_ARG_FROM, conv1UserWeightsMemory}, {DNNL_ARG_TO, conv1WeightsMemory}});
    }

    // memory for conv destination
    auto conv1DestMemory = memory(conv1PrimitiveDesc.dst_desc(), eng);

    // add convolution primitive to network forward
    netForward.push_back(convolution_forward(conv1PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_WEIGHTS, conv1WeightsMemory},
        {DNNL_ARG_BIAS, conv1BiasMemory},
        {DNNL_ARG_DST, conv1DestMemory}
    });

    // -------------------------------
    //     1st batch normalization
    // -------------------------------
    memory::dims batchNorm1Dims = {BATCH_SIZE, convFilters, convDestDims.at(2), convDestDims.at(3)};

    const float epsilon1 = 0.00001;
    auto batchNorm1PrimitiveDesc = batch_normalization_forward::primitive_desc(
        eng, prop_kind::forward, conv1PrimitiveDesc.dst_desc(), conv1PrimitiveDesc.dst_desc(), epsilon1, normalization_flags::none
    );

    auto batchNorm1DestMemory = memory(batchNorm1PrimitiveDesc.dst_desc(), eng);

    netForward.push_back(batch_normalization_forward(batchNorm1PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_SRC, conv1DestMemory},
        {DNNL_ARG_DST, batchNorm1DestMemory}
    });

    // -------------------------------
    //     1st RELU
    // -------------------------------
    memory::dims relu1Dims = {BATCH_SIZE, convFilters, convDestDims.at(2), convDestDims.at(3)};
    const float negativeSlope1 = 0.0f;

    auto relu1PrimitiveDesc = eltwise_forward::primitive_desc(
        eng, prop_kind::forward, algorithm::eltwise_relu,
        conv1PrimitiveDesc.dst_desc(), conv1PrimitiveDesc.dst_desc(), negativeSlope1);

    auto relu1DestMemory = memory(relu1PrimitiveDesc.dst_desc(), eng);

    netForward.push_back(eltwise_forward(relu1PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_SRC, batchNorm1DestMemory},
        {DNNL_ARG_DST, relu1DestMemory}
    });

    // -------------------------------
    //     2nd conv layer
    // -------------------------------
    // TODO copy memory from 1st layer dest

    // 1st conv produces tensors of dims (if not downsampling)
    // [batch, 3, 224] x [64, 3, 3, 3] -> [batch, 64, 224, 224]
    // 2nd conv produces
    // [batch, 64, 224, 224] x [64, 64, 3, 3] -> [batch, 64, 224, 224]

    memory::dims conv2SourceDims = {convSourceDims.at(0), convFilters, 
        downSample ? convSourceDims.at(2) / 2 : convSourceDims.at(2),
        downSample ? convSourceDims.at(3) / 2 : convSourceDims.at(3)};
    memory::dims conv2WeightsDims = {convFilters, convFilters, 3, 3};
    memory::dims conv2BiasDims = {convFilters};
    memory::dims conv2Strides = {1, 1};
    memory::dims conv2Padding = {1, 1};
    
    std::vector<float> conv2InitSourceMemory(product(conv2SourceDims));
    for (size_t i = 0; i < conv2InitSourceMemory.size(); ++i) {
        conv2InitSourceMemory[i] = sinf(float(i));
    }
    std::vector<float> conv2Weights(product(conv2WeightsDims));
    for (size_t i = 0; i < conv2Weights.size(); ++i) {
        conv2Weights[i] = sinf(float(i));
    }
    std::vector<float> conv2Bias(product(conv2BiasDims));
    for (size_t i = 0; i < conv2Bias.size(); ++i) {
        conv2Bias[i] = sinf(float(i));
    }

    // memory for tensor data
    auto conv2UserSourceMemory = 
        memory({{conv2SourceDims}, dt::f32, tag::nchw}, eng);
    auto conv2UserWeightsMemory =
        memory({{conv2WeightsDims}, dt::f32, tag::oihw}, eng);
    auto conv2BiasMemory =
        memory({{conv2BiasDims}, dt::f32, tag::x}, eng);
    
    write_to_dnnl_memory((void *) conv2InitSourceMemory.data(), conv2UserSourceMemory);
    write_to_dnnl_memory((void *) conv2Weights.data(), conv2UserWeightsMemory);
    write_to_dnnl_memory(conv2Bias.data(), conv2BiasMemory);

    // memory descriptors for convolution data
    auto conv2SourceMD = memory::desc({conv2SourceDims}, dt::f32, tag::any);
    auto conv2DestMD = memory::desc({convDestDims}, dt::f32, tag::any);
    auto conv2WeightsMD = memory::desc({conv2WeightsDims}, dt::f32, tag::any);
    auto conv2BiasMD = memory::desc({conv2BiasDims}, dt::f32, tag::any);

    // convolution primitive descriptor
    auto conv2PrimitiveDesc = convolution_forward::primitive_desc(
        eng, prop_kind::forward, algorithm::convolution_direct,
        conv2SourceMD, conv2WeightsMD, conv2BiasMD,
        conv2DestMD, conv2Strides, conv2Padding, conv2Padding
    );

    // reorder primitives between input and conv src if needed (??)
    auto conv2SourceMemory = conv2UserSourceMemory;
    if (conv2PrimitiveDesc.src_desc() != conv2UserSourceMemory.get_desc()) {
        conv2UserSourceMemory = memory(conv2PrimitiveDesc.src_desc(), eng);
        netForward.push_back(reorder(conv2UserSourceMemory, conv2SourceMemory));
        netForwardArgs.push_back({{DNNL_ARG_FROM, conv2UserSourceMemory}, {DNNL_ARG_TO, conv2SourceMemory}});
    }

    auto conv2WeightsMemory = conv2UserWeightsMemory;
    if (conv2PrimitiveDesc.weights_desc() != conv2UserWeightsMemory.get_desc()) {
        conv2WeightsMemory = memory(conv2PrimitiveDesc.weights_desc(), eng);
        netForward.push_back(reorder(conv2UserWeightsMemory, conv2WeightsMemory));
        netForwardArgs.push_back({{DNNL_ARG_FROM, conv2UserWeightsMemory}, {DNNL_ARG_TO, conv2WeightsMemory}});
    }

    // memory for conv destination
    auto conv2DestMemory = memory(conv2PrimitiveDesc.dst_desc(), eng);

    // add convolution primitive to network forward
    netForward.push_back(convolution_forward(conv2PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_WEIGHTS, conv2WeightsMemory},
        {DNNL_ARG_BIAS, conv2BiasMemory},
        {DNNL_ARG_DST, conv2DestMemory}
    });

    // -------------------------------
    //     2nd batch normalization
    // -------------------------------
    memory::dims batchNorm2Dims = {BATCH_SIZE, convFilters, convDestDims.at(2), convDestDims.at(3)};

    const float epsilon2 = 0.00001;
    auto batchNorm2PrimitiveDesc = batch_normalization_forward::primitive_desc(
        eng, prop_kind::forward, conv2PrimitiveDesc.dst_desc(), conv2PrimitiveDesc.dst_desc(), epsilon2, normalization_flags::none
    );

    auto batchNorm2DestMemory = memory(batchNorm2PrimitiveDesc.dst_desc(), eng);

    netForward.push_back(batch_normalization_forward(batchNorm2PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_SRC, conv2DestMemory},
        {DNNL_ARG_DST, batchNorm2DestMemory}
    });

    // -------------------------------
    //     2nd RELU
    // -------------------------------
    memory::dims relu2Dims = {BATCH_SIZE, convFilters, convDestDims.at(2), convDestDims.at(3)};
    const float negativeSlope2 = 0.0f;

    auto relu2PrimitiveDesc = eltwise_forward::primitive_desc(
        eng, prop_kind::forward, algorithm::eltwise_relu,
        conv2PrimitiveDesc.dst_desc(), conv2PrimitiveDesc.dst_desc(), negativeSlope2);

    auto relu2DestMemory = memory(relu2PrimitiveDesc.dst_desc(), eng);

    netForward.push_back(eltwise_forward(relu2PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_SRC, batchNorm2DestMemory},
        {DNNL_ARG_DST, relu2DestMemory}
    });


    // TODO add backward stream

    // check if we forgot something
    assert(netForward.size() == netForwardArgs.size() && "Something is missing in forward network stream");

    int trainingIterations = 1;
    while (trainingIterations)  {
        // forward
        for (size_t i = 0; i < netForward.size(); ++i) {
            netForward.at(i).execute(s, netForwardArgs.at(i));
        }
    }

    s.wait();
}


int main(int argc, char **argv) {
    return handle_example_errors(resblock, parse_engine_kind(argc, argv));
}
