#pragma once

#include "pwmf_format.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace playworld {

// ============================================================================
// IEEE 754 Half-Precision & Scalar Quantization Conversions
// ============================================================================

// FP16 IEEE 754 half-precision float to FP32 conversion
float fp16_to_fp32(uint16_t h) noexcept;

// FP32 to IEEE 754 half-precision FP16 conversion (round to nearest even)
uint16_t fp32_to_fp16(float f) noexcept;

// BF16 Brain Floating Point to FP32 conversion
inline float bf16_to_fp32(uint16_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// FP32 to BF16 conversion (round to nearest even)
inline uint16_t fp32_to_bf16(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    // Add rounding bias
    u += 0x00007FFF + ((u >> 16) & 1);
    return static_cast<uint16_t>(u >> 16);
}

// FP8 E4M3 (1 sign, 4 exp, 3 mantissa, bias 7) to FP32
float fp8_e4m3_to_fp32(uint8_t byte) noexcept;

// FP32 to FP8 E4M3
uint8_t fp32_to_fp8_e4m3(float f) noexcept;

// FP8 E5M2 (1 sign, 5 exp, 2 mantissa, bias 15) to FP32
float fp8_e5m2_to_fp32(uint8_t byte) noexcept;

// FP32 to FP8 E5M2
uint8_t fp32_to_fp8_e5m2(float f) noexcept;

// ============================================================================
// Block Dequantization & Quantization Kernels
// ============================================================================

// Dequantizes a single 20-byte INT4Block32 (32 weights) to 32 FP32 floats.
// Unpacks low nibble = qs[i] & 0x0F, high nibble = (qs[i] >> 4) & 0x0F.
// When subtract_bias is true: out[i] = (float(nibble) - bias) * scale.
// When subtract_bias is false: out[i] = float(nibble) * scale + bias.
void DequantizeINT4Block32(const INT4Block32* block, float* out, bool subtract_bias = true) noexcept;

// Vectorized batch dequantization for N blocks of INT4Block32 (uses ARM NEON when available)
void DequantizeINT4Block32Batch(const INT4Block32* blocks, size_t num_blocks, float* out, bool subtract_bias = true) noexcept;

// Quantize 32 FP32 floats into a single INT4Block32
void QuantizeINT4Block32(const float* in, INT4Block32* block) noexcept;
void QuantizeINT4Block32Batch(const float* in, size_t num_elements, INT4Block32* blocks) noexcept;

// INT8 symmetric dequantization: out[i] = int8(data[i]) * scale
void DequantizeINT8Symm(const int8_t* data, size_t num_elements, float scale, float* out) noexcept;
void QuantizeINT8Symm(const float* in, size_t num_elements, int8_t* out, float& out_scale) noexcept;

// FP8 dequantization
void DequantizeFP8_E4M3(const uint8_t* data, size_t num_elements, float* out) noexcept;
void DequantizeFP8_E5M2(const uint8_t* data, size_t num_elements, float* out) noexcept;

// FP16 / BF16 batch dequantization
void DequantizeFP16(const uint16_t* data, size_t num_elements, float* out) noexcept;
void DequantizeBF16(const uint16_t* data, size_t num_elements, float* out) noexcept;

// ============================================================================
// Tensor Class: Managing Shapes, Strides, Quantization, and Buffers
// ============================================================================

class Tensor {
public:
    Tensor();

    // Owning constructor: allocates its own memory buffer
    Tensor(std::string name, uint32_t ndims, const uint32_t* shape,
           QuantType quant_type, float global_scale = 1.0f);

    // Non-owning view constructor: points to external memory (e.g. mmap container)
    Tensor(std::string name, uint32_t ndims, const uint32_t* shape,
           QuantType quant_type, const uint8_t* data, size_t data_bytes,
           float global_scale = 1.0f);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;

    // Accessors
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] uint32_t ndims() const noexcept { return ndims_; }
    [[nodiscard]] const std::array<uint32_t, 5>& shape() const noexcept { return shape_; }
    [[nodiscard]] uint32_t dim(size_t i) const noexcept { return i < 5 ? shape_[i] : 1; }
    [[nodiscard]] const std::array<uint64_t, 5>& strides() const noexcept { return strides_; }
    [[nodiscard]] uint64_t numel() const noexcept { return numel_; }
    [[nodiscard]] QuantType quant_type() const noexcept { return quant_type_; }
    [[nodiscard]] float global_scale() const noexcept { return global_scale_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return data_bytes_; }
    [[nodiscard]] bool is_view() const noexcept { return is_view_; }

    [[nodiscard]] const uint8_t* data() const noexcept;
    [[nodiscard]] uint8_t* mutable_data();

    template <typename T>
    [[nodiscard]] const T* data_as() const noexcept {
        return reinterpret_cast<const T*>(data());
    }

    template <typename T>
    [[nodiscard]] T* mutable_data_as() {
        return reinterpret_cast<T*>(mutable_data());
    }

    // Dequantizes the tensor payload into an FP32 buffer
    [[nodiscard]] std::vector<float> DequantizeToFP32() const;
    void DequantizeTo(float* dst) const;

    // Static helper to compute payload size in bytes for given shape & quant_type
    static size_t CalculateDataBytes(uint32_t ndims, const uint32_t* shape, QuantType qt) noexcept;

private:
    void ComputeStridesAndNumel() noexcept;

    std::string name_;
    uint32_t ndims_{0};
    std::array<uint32_t, 5> shape_{0, 0, 0, 0, 0};
    std::array<uint64_t, 5> strides_{0, 0, 0, 0, 0};
    uint64_t numel_{0};
    QuantType quant_type_{QuantType::FP32};
    float global_scale_{1.0f};

    size_t data_bytes_{0};
    bool is_view_{false};
    const uint8_t* view_ptr_{nullptr};
    std::vector<uint8_t> owned_buffer_;
};

} // namespace playworld
