#pragma once

#include <cstdint>

class MultiBiquad {
public:
    enum class FilterType {
        // PascalCase names (modern)
        LowPass,
        HighPass,
        BandPass,
        BandStop,
        AllPass,
        Peak,
        LowShelf,
        HighShelf,
        // ALL_CAPS aliases for backward compatibility
        LOW_PASS   = LowPass,
        HIGH_PASS  = HighPass,
        BAND_PASS  = BandPass,
        BAND_STOP  = BandStop,
        ALL_PASS   = AllPass,
        PEAK       = Peak,
        LOW_SHELF  = LowShelf,
        HIGH_SHELF = HighShelf,
    };

    // Class-scope aliases for callers using MultiBiquad::LOW_PASS style
    static constexpr FilterType LOW_PASS   = FilterType::LowPass;
    static constexpr FilterType HIGH_PASS  = FilterType::HighPass;
    static constexpr FilterType BAND_PASS  = FilterType::BandPass;
    static constexpr FilterType BAND_STOP  = FilterType::BandStop;
    static constexpr FilterType ALL_PASS   = FilterType::AllPass;
    static constexpr FilterType PEAK       = FilterType::Peak;
    static constexpr FilterType LOW_SHELF  = FilterType::LowShelf;
    static constexpr FilterType HIGH_SHELF = FilterType::HighShelf;

    struct FilterParams {
        double   gain_db     = 0.0;
        double   frequency   = 0.0;
        uint32_t sample_rate = 44100;
        double   q_factor    = 0.717;
        bool     is_bandwidth = false;
    };

    MultiBiquad() noexcept;

    double ProcessSample(double sample) noexcept;
    void   Reset() noexcept;

    void RefreshFilter(FilterType type, const FilterParams& params);

    // Backward-compatible overload for callers not yet migrated
    void RefreshFilter(
        FilterType type,
        float gain_db,
        float frequency,
        uint32_t sample_rate,
        float q_factor,
        bool is_bandwidth
    );

private:
    double x1_ = 0.0;
    double x2_ = 0.0;
    double y1_ = 0.0;
    double y2_ = 0.0;
    double a1_ = 0.0;
    double a2_ = 0.0;
    double b0_ = 0.0;
    double b1_ = 0.0;
    double b2_ = 0.0;
};
