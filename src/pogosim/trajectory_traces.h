#ifndef TRAJECTORY_TRACES_H
#define TRAJECTORY_TRACES_H

#include <deque>
#include <memory>
#include <vector>

#include <SDL2/SDL.h>
#include <box2d/box2d.h>

enum class boundary_condition_t;
class PogobotObject;

class TrajectoryTraces {
public:
    void reset(std::vector<std::shared_ptr<PogobotObject>> const& robots);
    void update(std::vector<std::shared_ptr<PogobotObject>> const& robots);
    void render(SDL_Renderer* renderer,
            boundary_condition_t boundary_condition,
            float domain_w,
            float domain_h) const;

private:
    b2Vec2 get_robot_tail_position(PogobotObject const& robot) const;

    std::vector<std::deque<b2Vec2>> traces;
};

#endif // TRAJECTORY_TRACES_H
