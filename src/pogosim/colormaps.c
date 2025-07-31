
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "colormaps.h"


#define SCALE_0_255_TO_0_25(x) ((uint8_t)(((x) * (25.0f / 255.0f)) + 0.5f))

void hsv_to_rgb(float h, float s, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    float r_prime, g_prime, b_prime;
    
    if (h >= 0 && h < 60) {
        r_prime = c; g_prime = x; b_prime = 0;
    } else if (h >= 60 && h < 120) {
        r_prime = x; g_prime = c; b_prime = 0;
    } else if (h >= 120 && h < 180) {
        r_prime = 0; g_prime = c; b_prime = x;
    } else if (h >= 180 && h < 240) {
        r_prime = 0; g_prime = x; b_prime = c;
    } else if (h >= 240 && h < 300) {
        r_prime = x; g_prime = 0; b_prime = c;
    } else {
        r_prime = c; g_prime = 0; b_prime = x;
    }
    
    *r = (uint8_t)((r_prime + m) * 255.0f);
    *g = (uint8_t)((g_prime + m) * 255.0f);
    *b = (uint8_t)((b_prime + m) * 255.0f);
}

void qualitative_colormap(uint8_t value, uint8_t *r, uint8_t *g, uint8_t *b) {
    static const uint8_t colormap[][3] = {
        {25,  0,  0}, // red
        { 0, 25,  0}, // green
        { 0,  0, 25}, // blue
        {25, 25,  0}, // yellow
        {25,  0, 25}, // magenta
        { 0, 25, 25}, // cyan
        {13,  0, 13}, // purple
        {13, 13,  0}, // olive
        { 0, 13, 13}, // teal
        {10, 10, 10}  // gray
    };

    size_t const num_colors = sizeof colormap / sizeof *colormap;
    uint8_t index = value % num_colors;

    *r = colormap[index][0];
    *g = colormap[index][1];
    *b = colormap[index][2];
}

void rainbow_colormap(uint8_t value, uint8_t *r, uint8_t *g, uint8_t *b) {
    float const normalized = (float)value / 255.0f * 6.0f;
    int   const region     = (int)normalized;      // 0 … 5
    float const fraction   = normalized - region;  // 0 … 1

    uint8_t r_raw = 0, g_raw = 0, b_raw = 0;

    switch (region) {
    case 0: r_raw = 255;              g_raw = (uint8_t)(255 * fraction);  break;          // red → yellow
    case 1: r_raw = (uint8_t)(255 * (1.0f - fraction)); g_raw = 255;      break;          // yellow → green
    case 2: g_raw = 255;              b_raw = (uint8_t)(255 * fraction);  break;          // green → cyan
    case 3: g_raw = (uint8_t)(255 * (1.0f - fraction)); b_raw = 255;      break;          // cyan → blue
    case 4: r_raw = (uint8_t)(255 * fraction);       b_raw = 255;         break;          // blue → magenta
    case 5: r_raw = 255;              b_raw = (uint8_t)(255 * (1.0f - fraction)); break;  // magenta → red
    default: break;
    }

    // Down-scale so nothing exceeds 25
    *r = SCALE_0_255_TO_0_25(r_raw);
    *g = SCALE_0_255_TO_0_25(g_raw);
    *b = SCALE_0_255_TO_0_25(b_raw);

    // Ensure at least one channel is non-zero for visibility
    if (*r == 0 && *g == 0 && *b == 0) { *r = 1; }
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
