#ifndef RENDER_H
#define RENDER_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <box2d/box2d.h>

#include "geometry.h"

/**
 * @brief Draws a polygon using an SDL renderer.
 *
 * This function renders a polygon by drawing red lines between consecutive vertices on the provided SDL renderer.
 * It also closes the polygon by drawing a line between the last and first vertex.
 *
 * @param renderer Pointer to the SDL_Renderer to use for drawing.
 * @param polygon A vector of b2Vec2 points defining the polygon.
 */
void draw_polygon(SDL_Renderer* renderer, const std::vector<b2Vec2>& polygon);

/**
 * @brief Saves the content of an SDL window to a PNG file.
 *
 * This function reads the pixels from the SDL renderer associated with a window and saves the image as a PNG file.
 * It also ensures that the directory path for the output file exists.
 *
 * @param renderer Pointer to the SDL_Renderer used for capturing the window content.
 * @param window Pointer to the SDL_Window from which to capture the content.
 * @param filename The file path where the PNG image will be saved.
 */
void save_window_to_png(SDL_Renderer* renderer, SDL_Window* window, const std::string& filename);

#endif // RENDER_H

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
