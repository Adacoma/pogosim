#include <SDL2/SDL.h>
#include <box2d/box2d.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "distances.h"
#include "render.h"
#include "spogobot.h"
#include "fpng.h"


void draw_polygon(SDL_Renderer* renderer, const std::vector<b2Vec2>& polygon) {
    if (polygon.size() < 3) {
        std::cerr << "Error: Polygon must have at least 3 points to be drawable." << std::endl;
        return;
    }

    // Set the drawing color for the polygon
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red color with full opacity

    // Draw lines between consecutive points
    for (size_t i = 0; i < polygon.size() - 1; ++i) {
        auto const orig_pos = visualization_position(polygon[i].x, polygon[i].y);
        auto const dest_pos = visualization_position(polygon[i+1].x, polygon[i+1].y);
        SDL_RenderDrawLine(renderer,
                           static_cast<int>(orig_pos.x),
                           static_cast<int>(orig_pos.y),
                           static_cast<int>(dest_pos.x),
                           static_cast<int>(dest_pos.y));
    }

    // Connect the last point to the first to close the polygon
    auto const back_pos  = visualization_position(polygon.back().x, polygon.back().y);
    auto const front_pos = visualization_position(polygon.front().x, polygon.front().y);
    SDL_RenderDrawLine(renderer,
                       static_cast<int>(back_pos.x),
                       static_cast<int>(back_pos.y),
                       static_cast<int>(front_pos.x),
                       static_cast<int>(front_pos.y));
}


void save_window_to_png(SDL_Renderer* renderer, SDL_Window* window, const std::string& filename) {
    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    ensure_directories_exist(filename);

    std::vector<Uint8> pixels(width * height * 4); // RGBA format
    if (SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_RGBA32, pixels.data(), width * 4) != 0) {
        std::cerr << "Error reading pixels: " << SDL_GetError() << std::endl;
        return;
    }

//    // Flip the image vertically (SDL's origin is top-left, PNG's is bottom-left)
//    std::vector<Uint8> flippedPixels(width * height * 4);
//    for (int row = 0; row < height; ++row) {
//        std::copy_n(
//            &pixels[(height - row - 1) * width * 4], // Source row (flipped)
//            width * 4,                               // Row size
//            &flippedPixels[row * width * 4]          // Destination row
//        );
//    }

    // Save to PNG
    if (!fpng::fpng_encode_image_to_file(filename.c_str(), pixels.data(), width, height, 4)) {
        glogger->warn("Error writing PNG file '{}'", filename);
    } else {
        glogger->debug("Saved window content to '{}'", filename);
    }
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
