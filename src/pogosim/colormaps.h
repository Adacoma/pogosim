#ifndef COLORMAPS_H
#define COLORMAPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure representing a color with red, green, and blue components.
 */
typedef struct {
    uint8_t r; /**< Red component (0-255) */
    uint8_t g; /**< Green component (0-255) */
    uint8_t b; /**< Blue component (0-255) */
} color_t;


/**
 * @brief  HSV → RGB conversion helper
 *
 * This routine converts a colour expressed in **HSV** (Hue, Saturation,
 * Value) space to its equivalent **RGB** (Red, Green, Blue) representation
 * with 8-bit integer channels (0 – 255).  
 *
 * The implementation follows the classical algorithm described in
 * Foley & van Dam *Computer Graphics: Principles and Practice*  
 * (section “Converting HSV to RGB”), using the “hexcone” model:
 *
 * * Hue (@p h) is interpreted as an angle around the colour wheel,
 *   measured in **degrees** in \[0 … 360).  
 * * Saturation (@p s) and Value (@p v) are floating-point fractions
 *   in \[0.0 … 1.0], where **0** is fully desaturated (grey / black) and
 *   **1** is maximum chroma / brightness.  
 *
 * Internally, the algorithm:
 *
 * 1. Computes chroma *c = v·s*.  
 * 2. Finds the auxiliary value *x = c·(1 − |((h/60) mod 2) − 1|)* to locate
 *    the point inside the RGB cube for the current hue sector.  
 * 3. Adds the “match” term *m = v − c* to re-translate the temporary colour
 *    into the cube whose origin is black.  
 *
 * The six hue sectors \[0°,60°), \[60°,120°), …, \[300°,360°) are handled
 * explicitly to avoid expensive divisions at run-time (important on the
 * robot’s MCU).  The resulting floating-point RGB components are finally
 * scaled to integers in 0 … 255 and written back through the output
 * pointers.
 *
 * ### Typical use
 * @code{.c}
 * uint8_t r, g, b;
 * hsv_to_rgb(210.0f, 1.0f, 1.0f, &r, &g, &b); // vivid blue
 * pogobot_led_setColors(r, g, b, 0);           // light the main LED
 * @endcode
 *
 * ### Numerical accuracy
 * * The conversion preserves full saturation and value in the sense that
 *   at least one output channel will equal 255 whenever `v == 1.0f`
 *   **and** `s > 0`.  
 * * Rounding is done with a simple cast – values midway between two
 *   integers are rounded down (e.g. 254.7 → 254).  If you need
 *   round-to-nearest, add 0.5 f before the cast.  
 *
 * @param[in]  h  Hue angle in **degrees** (wraps outside \[0,360) without UB)
 * @param[in]  s  Saturation **fraction** in the closed interval \[0.0, 1.0]
 * @param[in]  v  Value / brightness **fraction** in \[0.0, 1.0]
 * @param[out] r  Pointer to destination for the **red**   component (0 – 255)
 * @param[out] g  Pointer to destination for the **green** component (0 – 255)
 * @param[out] b  Pointer to destination for the **blue**  component (0 – 255)
 *
 * @warning  The function assumes the output pointers are **non-NULL**.  
 *           Passing NULL is undefined behaviour.
 *
 * @see rainbow_colormap(), qualitative_colormap()
 */
void hsv_to_rgb(float h, float s, float v, uint8_t *r, uint8_t *g, uint8_t *b);


/**
 * @brief Maps a given value to a qualitative colormap.
 *
 * This function assigns a fixed color from a predefined qualitative colormap based on the
 * provided value. The colormap is defined as an array of 10 distinct colors. The input value is
 * mapped to a color index using modulo arithmetic, ensuring that it wraps around if the value
 * exceeds the number of available colors.
 *
 * @param value The input value used to select a color from the colormap.
 * @param r Pointer to a uint8_t variable where the red component of the selected color will be stored.
 * @param g Pointer to a uint8_t variable where the green component of the selected color will be stored.
 * @param b Pointer to a uint8_t variable where the blue component of the selected color will be stored.
 */
void qualitative_colormap(uint8_t const value, uint8_t *r, uint8_t *g, uint8_t *b);

/**
 * @brief Maps a given value to a rainbow colormap with smooth interpolation.
 *
 * This function computes a rainbow color based on the input value. The value, expected in the range
 * [0, 255], is normalized to a range [0, 6] to determine the color segment and the interpolation factor.
 * Depending on the segment, the function interpolates between two adjacent colors in the rainbow spectrum.
 *
 * The segments are as follows:
 * - Region 0: Red to Yellow
 * - Region 1: Yellow to Green
 * - Region 2: Green to Cyan
 * - Region 3: Cyan to Blue
 * - Region 4: Blue to Magenta
 * - Region 5: Magenta to Red
 *
 * @param value The input value (0-255) that determines the color.
 * @param r Pointer to a uint8_t variable where the red component of the computed color will be stored.
 * @param g Pointer to a uint8_t variable where the green component of the computed color will be stored.
 * @param b Pointer to a uint8_t variable where the blue component of the computed color will be stored.
 */
void rainbow_colormap(uint8_t const value, uint8_t *r, uint8_t *g, uint8_t *b);

#ifdef __cplusplus
}
#endif

#endif // COLORMAPS_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
