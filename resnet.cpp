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
    engine::kind engineKind,
    memory::dims convSourceDims,
    memory::dims convDestDims,
    int convFilters,
    bool downSample = true
) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = engine(engineKind, 0);

    stream s(eng);

    // TODO vvvv test purposes only
    convSourceDims = {BATCH_SIZE, 3, 224, 224};
    convDestDims = {BATCH_SIZE, convFilters, 112, 112};
    // TODO ^^^^ 

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

    const float epsilon = 0.00001;
    auto batchNorm1PrimitiveDesc = batch_normalization_forward::primitive_desc(
        eng, prop_kind::forward, conv1PrimitiveDesc.dst_desc(), conv1PrimitiveDesc.dst_desc(), epsilon, normalization_flags::none
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
    const float negativeSlope = 0.0f;

    auto relu1PrimitiveDesc = eltwise_forward::primitive_desc(
        eng, prop_kind::forward, algorithm::eltwise_relu,
        conv1PrimitiveDesc.dst_desc(), conv1PrimitiveDesc.dst_desc(), negativeSlope);

    auto relu1DestMemory = memory(relu1PrimitiveDesc.dst_desc(), eng);

    netForward.push_back(eltwise_forward(relu1PrimitiveDesc));
    netForwardArgs.push_back({
        {DNNL_ARG_SRC, batchNorm1DestMemory},
        {DNNL_ARG_DST, relu1DestMemory}
    });





}
