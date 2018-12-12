#pragma once
#include <ppl.h>
using namespace std;

#define Method __forceinline static

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static class SIMD_128
{
public:
    Method float ToFloat(__m128 a)
    {
        float sum;
        __m128 vsum;

        vsum = _mm_hadd_ps(a, a);
        vsum = _mm_hadd_ps(vsum, vsum);
        _mm_store_ss(&sum, vsum);
        return sum;
    };

    Method __m128 AddFloats(__m128 a, __m128 b)
    {
        return _mm_add_ps(a, b);
    };

    Method __m128i AddInt(__m128i a, __m128i b)
    {
        return _mm_add_epi32(a, b);
    };

    Method __m128 MultiplyFloats(__m128 a, __m128 b)
    {
        return _mm_mul_ps(a, b);
    };

    Method __m128i MultiplyInt(__m128i a, __m128i b)
    {
        return _mm_mul_epi32(a, b);
    };

    Method __m128 LoadFloat4(float a, float b, float c, float d)
    {
        return _mm_setr_ps(a, b, c, d);
    }

    Method __m128i LoadInt4(int a, int b, int c, int d)
    {
        return _mm_setr_epi32(a, b, c, d);
    }

    Method __m128 LoadFloat4(const float& start)
    {
        return _mm_load_ps(&start);
    };

    Method __m128i LoadInt4(const int& start)
    {
        return _mm_load_si128((__m128i*)start);
    };

    Method __m128 LoadFloat3(const float& start)
    {
        // load x, y with a 64 bit integer load (00YX)
        __m128i xy = _mm_loadl_epi64((const __m128i*)&start);

        // now load the z element using a 32 bit float load (000Z)
        __m128 z = _mm_load_ss(&start + 2);

        // we now need to cast the __m128i register into a __m128 one (0ZYX)
        return _mm_movelh_ps(_mm_castsi128_ps(xy), z);
    };

    Method __m128 LoadFloat2(const float& start)
    {
        // load x, y with a 64 bit integer load (00YX)
        __m128i xy = _mm_loadl_epi64((const __m128i*)&start);

        return _mm_castsi128_ps(xy);
    };

    Method __m128 LoadFloat2x2(const float& start, const float& start2)
    {
        // load ab with a 64 bit integer load (00BA)
        __m128i ab = _mm_loadl_epi64((const __m128i*)&start);

        // load dc with a 64 bit integer load (00CD)
        __m128i dc = _mm_loadl_epi64((const __m128i*)&start2);

        // (CDBA)
        return _mm_movelh_ps(_mm_castsi128_ps(ab), _mm_castsi128_ps(dc));
    };

    Method __m128 LoadFloat(float a)
    {
        return _mm_set_ss(a);
    };
};

class SIMD_256
{
public:
    Method float ToFloat(__m256 a)
    {
        __m256 t1 = _mm256_hadd_ps(a, a);
        __m256 t2 = _mm256_hadd_ps(t1, t1);
        __m128 t3 = _mm256_extractf128_ps(t2, 1);
        __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
        return _mm_cvtss_f32(t4);
    }

    Method __m256 AddFloats(__m256 a, __m256 b)
    {
        return _mm256_add_ps(a, b);
    };

    Method __m256 MultiplyFloats(__m256 a, __m256 b)
    {
        return _mm256_mul_ps(a, b);
    };

    Method __m256 LoadFloat8(float a, float b, float c, float d, float e, float f, float g, float h)
    {
        return _mm256_setr_ps(a, b, c, d, e, f, g, h);
    }

    Method __m256 LoadFloat8(const float& start)
    {
        return _mm256_load_ps(&start);
    };

    Method __m256 From128(__m128 a, __m128 b)
    {
        __m256 c = _mm256_insertf128_ps(c, a, 0);
        c = _mm256_insertf128_ps(c, b, 1);
        return c;
    }
};