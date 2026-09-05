#include "playworld/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace playworld {

// ============================================================================
// IEEE 754 Half-Precision & Scalar Quantization Conversions
// ============================================================================

float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign;
            float f = 0.0f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }
        // Subnormal: normalize mantissa
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
        const uint32_t biased_exp = (127 - 15) + exp;
        const uint32_t u = sign | (biased_exp << 23) | (mant << 13);
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    } else if (exp == 31) {
        // Infinity or NaN
        const uint32_t u = sign | 0x7F800000U | (mant << 13);
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }

    // Normalized
    const uint32_t biased_exp = exp + (127 - 15);
    const uint32_t u = sign | (biased_exp << 23) | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t fp32_to_fp16(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000;
    const int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x00800000U) >> (1 - exp);
        // Round to nearest even
        if ((mant & 0x1000) != 0) {
            mant += 0x2000;
        }
        return static_cast<uint16_t>(sign | (mant >> 13));
    } else if (exp >= 31) {
        if (exp == 143 && mant != 0) {
            return static_cast<uint16_t>(sign | 0x7E00 | (mant >> 13)); // NaN
        }
        return static_cast<uint16_t>(sign | 0x7C00); // Inf
    }

    // Round to nearest even
    if ((mant & 0x1000) != 0) {
        mant += 0x2000;
        if ((mant & 0x00800000U) != 0) {
            mant = 0;
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float fp8_e4m3_to_fp32(uint8_t byte) noexcept {
    const uint32_t sign = (byte & 0x80) ? 0x80000000U : 0U;
    const uint32_t exp  = (byte >> 3) & 0x0F;
    const uint32_t mant = byte & 0x07;

    // 0x7F and 0xFF represent NaN in E4M3
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        if (mant == 0) {
            float zero = 0.0f;
            if (sign) zero = -zero;
            return zero;
        }
        // Subnormal: 2^(-6) * (mant / 8.0)
        const float val = 0.015625f * (static_cast<float>(mant) / 8.0f);
        return sign ? -val : val;
    }

    // Normal: 2^(exp - 7) * (1.0 + mant / 8.0)
    const float scale = std::ldexp(1.0f, static_cast<int>(exp) - 7);
    const float val = scale * (1.0f + static_cast<float>(mant) / 8.0f);
    return sign ? -val : val;
}

uint8_t fp32_to_fp8_e4m3(float f) noexcept {
    if (std::isnan(f)) {
        return 0x7F; // Standard NaN
    }
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint8_t sign = (u & 0x80000000U) ? 0x80 : 0x00;
    const float abs_f = std::fabs(f);

    // Max finite in E4M3 is 448.0
    if (abs_f >= 448.0f) {
        return sign | 0x7E; // Max finite (exp=15, mant=6)
    }

    // Subnormals: abs_f < 2^(-6) = 0.015625f
    if (abs_f < 0.015625f) {
        int mant = static_cast<int>(std::round(abs_f * 512.0f));
        if (mant > 7) mant = 7;
        return sign | static_cast<uint8_t>(mant);
    }

    int exp = 0;
    float norm = std::frexp(abs_f, &exp);
    norm *= 2.0f;
    exp -= 1;
    int biased_exp = exp + 7;
    if (biased_exp < 1) biased_exp = 1;
    if (biased_exp > 15) biased_exp = 15;

    int mant = static_cast<int>(std::round((norm - 1.0f) * 8.0f));
    if (mant >= 8) {
        mant = 0;
        biased_exp++;
        if (biased_exp > 15) {
            biased_exp = 15;
            mant = 6;
        }
    }
    if (biased_exp == 15 && mant >= 7) {
        mant = 6; // Avoid 0x7F (NaN)
    }
    return sign | static_cast<uint8_t>((biased_exp << 3) | (mant & 0x07));
}

float fp8_e5m2_to_fp32(uint8_t byte) noexcept {
    const uint32_t sign = (byte & 0x80) ? 0x80000000U : 0U;
    const uint32_t exp  = (byte >> 2) & 0x1F;
    const uint32_t mant = byte & 0x03;

    if (exp == 0x1F) {
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        if (mant == 0) {
            float zero = 0.0f;
            return sign ? -zero : zero;
        }
        // Subnormal: 2^(-14) * (mant / 4.0)
        const float val = std::ldexp(1.0f, -14) * (static_cast<float>(mant) / 4.0f);
        return sign ? -val : val;
    }

    // Normal: 2^(exp - 15) * (1.0 + mant / 4.0)
    const float scale = std::ldexp(1.0f, static_cast<int>(exp) - 15);
    const float val = scale * (1.0f + static_cast<float>(mant) / 4.0f);
    return sign ? -val : val;
}

uint8_t fp32_to_fp8_e5m2(float f) noexcept {
    if (std::isnan(f)) {
        return 0x7F;
    }
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint8_t sign = (u & 0x80000000U) ? 0x80 : 0x00;
    if (std::isinf(f)) {
        return sign | 0x7C;
    }
    const float abs_f = std::fabs(f);
    if (abs_f >= 57344.0f) {
        return sign | 0x7C; // Overflow to Inf
    }
    const float min_norm = std::ldexp(1.0f, -14);
    if (abs_f < min_norm) {
        int mant = static_cast<int>(std::round(abs_f / min_norm * 4.0f));
        if (mant > 3) mant = 3;
        return sign | static_cast<uint8_t>(mant);
    }
    int exp = 0;
    float norm = std::frexp(abs_f, &exp);
    norm *= 2.0f;
    exp -= 1;
    int biased_exp = exp + 15;
    if (biased_exp < 1) biased_exp = 1;
    if (biased_exp > 30) biased_exp = 30;

    int mant = static_cast<int>(std::round((norm - 1.0f) * 4.0f));
    if (mant >= 4) {
        mant = 0;
        biased_exp++;
        if (biased_exp >= 31) {
            return sign | 0x7C;
        }
    }
    return sign | static_cast<uint8_t>((biased_exp << 2) | (mant & 0x03));
}

// ============================================================================
// Block Dequantization & Quantization Implementations
// ============================================================================

void DequantizeINT4Block32(const INT4Block32* block, float* out, bool subtract_bias) noexcept {
    if (!block || !out) return;

    const float scale = fp16_to_fp32(block->scale_fp16);
    const float bias  = fp16_to_fp32(block->bias_fp16);

#if defined(__ARM_NEON)
    const uint8x16_t raw = vld1q_u8(block->qs);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x16_t low = vandq_u8(raw, mask);
    const uint8x16_t high = vshrq_n_u8(raw, 4);

    // Interleave low and high nibbles: low0, high0, low1, high1...
    const uint8x16x2_t zip = vzipq_u8(low, high);

    const float32x4_t v_scale = vdupq_n_f32(scale);
    const float32x4_t v_bias  = vdupq_n_f32(bias);

    // Process zip.val[0] (nibbles 0..15) and zip.val[1] (nibbles 16..31)
    for (int part = 0; part < 2; ++part) {
        const uint8x16_t part_u8 = (part == 0) ? zip.val[0] : zip.val[1];
        const uint16x8_t u16_low  = vmovl_u8(vget_low_u8(part_u8));
        const uint16x8_t u16_high = vmovl_u8(vget_high_u8(part_u8));

        const uint32x4_t u32_0 = vmovl_u16(vget_low_u16(u16_low));
        const uint32x4_t u32_1 = vmovl_u16(vget_high_u16(u16_low));
        const uint32x4_t u32_2 = vmovl_u16(vget_low_u16(u16_high));
        const uint32x4_t u32_3 = vmovl_u16(vget_high_u16(u16_high));

        const float32x4_t f0 = vcvtq_f32_u32(u32_0);
        const float32x4_t f1 = vcvtq_f32_u32(u32_1);
        const float32x4_t f2 = vcvtq_f32_u32(u32_2);
        const float32x4_t f3 = vcvtq_f32_u32(u32_3);

        const int offset = part * 16;
        if (subtract_bias) {
            vst1q_f32(out + offset + 0,  vmulq_f32(vsubq_f32(f0, v_bias), v_scale));
            vst1q_f32(out + offset + 4,  vmulq_f32(vsubq_f32(f1, v_bias), v_scale));
            vst1q_f32(out + offset + 8,  vmulq_f32(vsubq_f32(f2, v_bias), v_scale));
            vst1q_f32(out + offset + 12, vmulq_f32(vsubq_f32(f3, v_bias), v_scale));
        } else {
            vst1q_f32(out + offset + 0,  vmlaq_f32(v_bias, f0, v_scale));
            vst1q_f32(out + offset + 4,  vmlaq_f32(v_bias, f1, v_scale));
            vst1q_f32(out + offset + 8,  vmlaq_f32(v_bias, f2, v_scale));
            vst1q_f32(out + offset + 12, vmlaq_f32(v_bias, f3, v_scale));
        }
    }
#else
    for (size_t i = 0; i < 16; ++i) {
        const uint8_t byte = block->qs[i];
        const uint8_t q_low  = byte & 0x0F;
        const uint8_t q_high = (byte >> 4) & 0x0F;

        if (subtract_bias) {
            out[2 * i]     = (static_cast<float>(q_low)  - bias) * scale;
            out[2 * i + 1] = (static_cast<float>(q_high) - bias) * scale;
        } else {
            out[2 * i]     = static_cast<float>(q_low)  * scale + bias;
            out[2 * i + 1] = static_cast<float>(q_high) * scale + bias;
        }
    }
#endif
}

void DequantizeINT4Block32Batch(const INT4Block32* blocks, size_t num_blocks, float* out, bool subtract_bias) noexcept {
    if (!blocks || !out) return;
    for (size_t b = 0; b < num_blocks; ++b) {
        DequantizeINT4Block32(&blocks[b], out + b * 32, subtract_bias);
    }
}

void QuantizeINT4Block32(const float* in, INT4Block32* block) noexcept {
    if (!in || !block) return;

    float min_val = in[0];
    float max_val = in[0];
    for (size_t i = 1; i < 32; ++i) {
        if (in[i] < min_val) min_val = in[i];
        if (in[i] > max_val) max_val = in[i];
    }

    float range = max_val - min_val;
    float scale = range / 15.0f;
    float bias = 0.0f;

    if (range < 1e-7f) {
        scale = 1.0f;
        bias = -min_val;
        bias = std::clamp(bias, -65504.0f, 65504.0f);
    } else {
        bias = -min_val / scale;
        bias = std::clamp(bias, -65504.0f, 65504.0f);
    }

    block->scale_fp16 = fp32_to_fp16(scale);
    block->bias_fp16  = fp32_to_fp16(bias);

    // Quantize 32 elements into 16 bytes
    for (size_t i = 0; i < 16; ++i) {
        float f0 = in[2 * i];
        float f1 = in[2 * i + 1];

        int q0 = static_cast<int>(std::round(f0 / scale + bias));
        int q1 = static_cast<int>(std::round(f1 / scale + bias));

        q0 = std::clamp(q0, 0, 15);
        q1 = std::clamp(q1, 0, 15);

        block->qs[i] = static_cast<uint8_t>((q1 << 4) | (q0 & 0x0F));
    }
}

void QuantizeINT4Block32Batch(const float* in, size_t num_elements, INT4Block32* blocks) noexcept {
    if (!in || !blocks) return;
    const size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        QuantizeINT4Block32(in + b * 32, &blocks[b]);
    }
}

void DequantizeINT8Symm(const int8_t* data, size_t num_elements, float scale, float* out) noexcept {
    if (!data || !out) return;

#if defined(__ARM_NEON)
    const float32x4_t v_scale = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 16 <= num_elements; i += 16) {
        const int8x16_t raw = vld1q_s8(data + i);
        const int16x8_t s16_low  = vmovl_s8(vget_low_s8(raw));
        const int16x8_t s16_high = vmovl_s8(vget_high_s8(raw));

        const int32x4_t s32_0 = vmovl_s16(vget_low_s16(s16_low));
        const int32x4_t s32_1 = vmovl_s16(vget_high_s16(s16_low));
        const int32x4_t s32_2 = vmovl_s16(vget_low_s16(s16_high));
        const int32x4_t s32_3 = vmovl_s16(vget_high_s16(s16_high));

        vst1q_f32(out + i + 0,  vmulq_f32(vcvtq_f32_s32(s32_0), v_scale));
        vst1q_f32(out + i + 4,  vmulq_f32(vcvtq_f32_s32(s32_1), v_scale));
        vst1q_f32(out + i + 8,  vmulq_f32(vcvtq_f32_s32(s32_2), v_scale));
        vst1q_f32(out + i + 12, vmulq_f32(vcvtq_f32_s32(s32_3), v_scale));
    }
    for (; i < num_elements; ++i) {
        out[i] = static_cast<float>(data[i]) * scale;
    }
#else
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = static_cast<float>(data[i]) * scale;
    }
#endif
}

void QuantizeINT8Symm(const float* in, size_t num_elements, int8_t* out, float& out_scale) noexcept {
    if (!in || !out || num_elements == 0) return;

    float max_abs = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        const float a = std::fabs(in[i]);
        if (a > max_abs) max_abs = a;
    }

    out_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / out_scale;

    for (size_t i = 0; i < num_elements; ++i) {
        int val = static_cast<int>(std::round(in[i] * inv_scale));
        val = std::clamp(val, -128, 127);
        out[i] = static_cast<int8_t>(val);
    }
}

void DequantizeFP8_E4M3(const uint8_t* data, size_t num_elements, float* out) noexcept {
    if (!data || !out) return;
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = fp8_e4m3_to_fp32(data[i]);
    }
}

void DequantizeFP8_E5M2(const uint8_t* data, size_t num_elements, float* out) noexcept {
    if (!data || !out) return;
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = fp8_e5m2_to_fp32(data[i]);
    }
}

void DequantizeFP16(const uint16_t* data, size_t num_elements, float* out) noexcept {
    if (!data || !out) return;
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = fp16_to_fp32(data[i]);
    }
}

void DequantizeBF16(const uint16_t* data, size_t num_elements, float* out) noexcept {
    if (!data || !out) return;
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = bf16_to_fp32(data[i]);
    }
}

// ============================================================================
// Tensor Class Implementation
// ============================================================================

Tensor::Tensor()
    : ndims_(0), numel_(0), quant_type_(QuantType::FP32), global_scale_(1.0f),
      data_bytes_(0), is_view_(false), view_ptr_(nullptr) {
    shape_.fill(0);
    strides_.fill(0);
}

Tensor::Tensor(std::string name, uint32_t ndims, const uint32_t* shape,
               QuantType quant_type, float global_scale)
    : name_(std::move(name)), ndims_(ndims), quant_type_(quant_type),
      global_scale_(global_scale), is_view_(false), view_ptr_(nullptr) {
    shape_.fill(0);
    if (shape && ndims > 0) {
        const uint32_t count = std::min(ndims, 5U);
        for (uint32_t i = 0; i < count; ++i) {
            shape_[i] = shape[i];
        }
    }
    ComputeStridesAndNumel();
    data_bytes_ = CalculateDataBytes(ndims_, shape_.data(), quant_type_);
    owned_buffer_.resize(data_bytes_, 0);
}

Tensor::Tensor(std::string name, uint32_t ndims, const uint32_t* shape,
               QuantType quant_type, const uint8_t* data, size_t data_bytes,
               float global_scale)
    : name_(std::move(name)), ndims_(ndims), quant_type_(quant_type),
      global_scale_(global_scale), data_bytes_(data_bytes),
      is_view_(true), view_ptr_(data) {
    shape_.fill(0);
    if (shape && ndims > 0) {
        const uint32_t count = std::min(ndims, 5U);
        for (uint32_t i = 0; i < count; ++i) {
            shape_[i] = shape[i];
        }
    }
    ComputeStridesAndNumel();
}

Tensor::Tensor(const Tensor& other)
    : name_(other.name_), ndims_(other.ndims_), shape_(other.shape_),
      strides_(other.strides_), numel_(other.numel_),
      quant_type_(other.quant_type_), global_scale_(other.global_scale_),
      data_bytes_(other.data_bytes_), is_view_(other.is_view_),
      view_ptr_(other.view_ptr_), owned_buffer_(other.owned_buffer_) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        name_ = other.name_;
        ndims_ = other.ndims_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        numel_ = other.numel_;
        quant_type_ = other.quant_type_;
        global_scale_ = other.global_scale_;
        data_bytes_ = other.data_bytes_;
        is_view_ = other.is_view_;
        view_ptr_ = other.view_ptr_;
        owned_buffer_ = other.owned_buffer_;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : name_(std::move(other.name_)), ndims_(other.ndims_),
      shape_(other.shape_), strides_(other.strides_), numel_(other.numel_),
      quant_type_(other.quant_type_), global_scale_(other.global_scale_),
      data_bytes_(other.data_bytes_), is_view_(other.is_view_),
      view_ptr_(other.view_ptr_), owned_buffer_(std::move(other.owned_buffer_)) {
    other.ndims_ = 0;
    other.numel_ = 0;
    other.data_bytes_ = 0;
    other.is_view_ = false;
    other.view_ptr_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        name_ = std::move(other.name_);
        ndims_ = other.ndims_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        numel_ = other.numel_;
        quant_type_ = other.quant_type_;
        global_scale_ = other.global_scale_;
        data_bytes_ = other.data_bytes_;
        is_view_ = other.is_view_;
        view_ptr_ = other.view_ptr_;
        owned_buffer_ = std::move(other.owned_buffer_);

        other.ndims_ = 0;
        other.numel_ = 0;
        other.data_bytes_ = 0;
        other.is_view_ = false;
        other.view_ptr_ = nullptr;
    }
    return *this;
}

const uint8_t* Tensor::data() const noexcept {
    return is_view_ ? view_ptr_ : owned_buffer_.data();
}

uint8_t* Tensor::mutable_data() {
    if (is_view_) {
        throw std::runtime_error("Cannot modify read-only memory-mapped tensor view");
    }
    return owned_buffer_.data();
}

void Tensor::ComputeStridesAndNumel() noexcept {
    if (ndims_ == 0) {
        numel_ = 0;
        strides_.fill(0);
        return;
    }
    numel_ = 1;
    for (uint32_t i = 0; i < ndims_; ++i) {
        numel_ *= shape_[i];
    }
    uint64_t stride = 1;
    for (int i = static_cast<int>(ndims_) - 1; i >= 0; --i) {
        strides_[static_cast<size_t>(i)] = stride;
        stride *= shape_[static_cast<size_t>(i)];
    }
    for (uint32_t i = ndims_; i < 5; ++i) {
        strides_[i] = 0;
    }
}

size_t Tensor::CalculateDataBytes(uint32_t ndims, const uint32_t* shape, QuantType qt) noexcept {
    if (ndims == 0 || !shape) return 0;
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (shape[i] == 0) return 0;
        n *= shape[i];
    }
    switch (qt) {
        case QuantType::FP32:
            return static_cast<size_t>(n * 4);
        case QuantType::FP16:
        case QuantType::BF16:
            return static_cast<size_t>(n * 2);
        case QuantType::INT8_SYMM:
        case QuantType::FP8_E4M3:
        case QuantType::FP8_E5M2:
            return static_cast<size_t>(n);
        case QuantType::INT4_BLOCK32: {
            const uint64_t num_blocks = (n + 31) / 32;
            return static_cast<size_t>(num_blocks * sizeof(INT4Block32));
        }
        case QuantType::INT4_BLOCK64: {
            const uint64_t num_blocks = (n + 63) / 64;
            return static_cast<size_t>(num_blocks * sizeof(INT4Block64));
        }
        default:
            return static_cast<size_t>(n * 4);
    }
}

std::vector<float> Tensor::DequantizeToFP32() const {
    std::vector<float> result(numel_, 0.0f);
    DequantizeTo(result.data());
    return result;
}

void Tensor::DequantizeTo(float* dst) const {
    if (!dst || numel_ == 0) return;
    const uint8_t* src = data();
    if (!src) return;

    switch (quant_type_) {
        case QuantType::FP32:
            std::memcpy(dst, src, numel_ * sizeof(float));
            break;
        case QuantType::FP16:
            DequantizeFP16(reinterpret_cast<const uint16_t*>(src), numel_, dst);
            break;
        case QuantType::BF16:
            DequantizeBF16(reinterpret_cast<const uint16_t*>(src), numel_, dst);
            break;
        case QuantType::INT8_SYMM:
            DequantizeINT8Symm(reinterpret_cast<const int8_t*>(src), numel_, global_scale_, dst);
            break;
        case QuantType::INT4_BLOCK32: {
            const size_t num_blocks = (numel_ + 31) / 32;
            DequantizeINT4Block32Batch(reinterpret_cast<const INT4Block32*>(src), num_blocks, dst, true);
            break;
        }
        case QuantType::FP8_E4M3:
            DequantizeFP8_E4M3(src, numel_, dst);
            break;
        case QuantType::FP8_E5M2:
            DequantizeFP8_E5M2(src, numel_, dst);
            break;
        default:
            std::memcpy(dst, src, std::min(data_bytes_, static_cast<size_t>(numel_ * sizeof(float))));
            break;
    }
}

} // namespace playworld
