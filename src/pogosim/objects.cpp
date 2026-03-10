
#include "utils.h"
#include "objects.h"
#include "robot.h"
#include "distances.h"
#include "simulator.h"
#include "objects_geometry.h"
#include "lights.h"

#include <cmath>
#include "SDL2_gfxPrimitives.h"


/************* OBJECT *************/ // {{{1

Object::Object(float _x, float _y, ObjectGeometry& _geom, std::string const& _category)
        : x(_x), y(_y), category(_category), geom(&_geom) {
    // ...
}

Object::Object(Simulation* simulation, float _x, float _y, Configuration const& config, std::string const& _category)
        : x(_x), y(_y), category(_category) {
    parse_configuration(config, simulation);
}

// XXX : destroy geom ??
Object::~Object() { }

void Object::launch_user_step([[maybe_unused]] float t) {
    // ...
}

void Object::parse_configuration(Configuration const& config, Simulation* simulation) {
    x = config["x"].get(x);
    y = config["y"].get(y);

    // Initialize geometry
    geom = object_geometry_factory(config, simulation); // XXX never destroyed
}

void Object::move(float _x, float _y, float _theta) {
    x = _x;
    y = _y;
    if (!std::isnan(_theta))
        theta = _theta;
}

arena_polygons_t Object::generate_contours(std::size_t points_per_contour) const {
    return geom->generate_contours(points_per_contour, {x, y});
}



/************* PhysicalObject *************/ // {{{1

PhysicalObject::PhysicalObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       std::string const& _category)
    : Object(_x, _y, geom, _category),
      id(_id),
      linear_damping(_linear_damping),
      angular_damping(_angular_damping),
      density(_density),
      friction(_friction),
      restitution(_restitution) { }

PhysicalObject::PhysicalObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       Configuration const& config,
       std::string const& _category)
    : Object(simulation, _x, _y, config, _category), id(_id) {
    parse_configuration(config, simulation);
}

void PhysicalObject::do_init([[maybe_unused]] b2WorldId world_id) {
    create_body(world_id);
}

void PhysicalObject::launch_user_step(float t) {
    Object::launch_user_step(t);

    // Compute acceleration statistics
    if (b2Body_IsValid(body_id)) {
        _estimated_dt = t - _last_time;
        _last_time = t;
        // Translational acceleration in world frame
        b2Vec2 now_v = b2Body_GetLinearVelocity(body_id);
        b2Vec2 a_world = (now_v - _prev_v) * (1.0f / _estimated_dt);
        _prev_v = now_v;
        // Specific force (proper accel) → subtract gravity
        b2Vec2 gravity = {0.0f, 0.0f};   // Same as Box2D world. TODO update
        b2Vec2 f_world = a_world - gravity;
        // Rotate into body frame
        _lin_acc = b2Body_GetLocalVector(body_id, f_world);
    }
}

b2Vec2 PhysicalObject::get_position() const {
    if (b2Body_IsValid(body_id)) {
        return b2Body_GetPosition(body_id);
    } else {
        return {NAN, NAN};
    }
}

float PhysicalObject::get_angle() const {
    if (b2Body_IsValid(body_id)) {
        b2Rot const rotation = b2Body_GetRotation(body_id);
        return std::atan2(rotation.s, rotation.c);
    } else {
        return NAN;
    }
}

float PhysicalObject::get_angular_velocity() const {
    if (b2Body_IsValid(body_id)) {
        return b2Body_GetAngularVelocity(body_id);
    } else {
        return NAN;
    }
}

b2Vec2 PhysicalObject::get_linear_acceleration() const {
    return _lin_acc;
}

void PhysicalObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    Object::parse_configuration(config, simulation);
    linear_damping = config["body_linear_damping"].get(0.0f);
    angular_damping = config["body_angular_damping"].get(0.0f);
    density = config["body_density"].get(10.0f);
    friction = config["body_friction"].get(0.3f);
    restitution = config["body_restitution"].get(0.5f);
}

void PhysicalObject::create_body(b2WorldId world_id) {
    // Create the body definition.
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = { x / VISUALIZATION_SCALE, y / VISUALIZATION_SCALE };
    bodyDef.linearDamping = linear_damping;
    bodyDef.angularDamping = angular_damping;
    bodyDef.isBullet = false;
    body_id = b2CreateBody(world_id, &bodyDef);

    // Set up a shape definition with common physical properties.
    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = density;
    shapeDef.friction = friction;
    shapeDef.restitution = restitution;
    shapeDef.enablePreSolveEvents = true;

    // Create shape
    geom->create_box2d_shape(body_id, shapeDef);

    // Assign an initial velocity.
    b2Vec2 velocity = { 0.0f, 0.0f };
    b2Body_SetLinearVelocity(body_id, velocity);
    b2Body_SetAngularVelocity(body_id, 0.0f);
    //b2Body_SetAwake(body_id, false);
}

void PhysicalObject::move(float _x, float _y, float _theta) {
    Object::move(_x, _y, _theta);
    if (b2Body_IsValid(body_id)) {
        b2Vec2 position = {_x / VISUALIZATION_SCALE, _y / VISUALIZATION_SCALE};
        b2Rot rotation = {sinf(_theta), cosf(_theta)};
        if (std::isnan(_theta)) {
            rotation = b2Body_GetRotation(body_id);
        }
        b2Body_SetTransform(body_id, position, rotation);
    }
}

arena_polygons_t PhysicalObject::generate_contours(std::size_t points_per_contour) const {
    return geom->generate_contours(points_per_contour, get_position());
}



/************* PassiveObject *************/ // {{{1

PassiveObject::PassiveObject(uint16_t _id, float _x, float _y,
       ObjectGeometry& geom,
       float _linear_damping, float _angular_damping,
       float _density, float _friction, float _restitution,
       std::string _colormap,
       std::string const& _category)
    : PhysicalObject(_id, _x, _y, geom,
      _linear_damping, _angular_damping,
      _density, _friction, _restitution, _category),
      colormap(_colormap) {
    // ...
}

PassiveObject::PassiveObject(Simulation* simulation, uint16_t _id, float _x, float _y,
       Configuration const& config,
       std::string const& _category)
    : PhysicalObject(simulation, _id, _x, _y, config, _category) {
    parse_configuration(config, simulation);
    // ...
}

void PassiveObject::render(SDL_Renderer* renderer, b2WorldId world_id) const {
    // Get object's position in the physics world
    b2Vec2 body_position = b2Body_GetPosition(body_id);

    // Identify object X and Y coordinates in visualization instance
    float screen_x = body_position.x * VISUALIZATION_SCALE;
    float screen_y = body_position.y * VISUALIZATION_SCALE;
    auto const pos = visualization_position(screen_x, screen_y);

    // Assign color based on object initial position
    uint8_t const value = (static_cast<int32_t>(x) + static_cast<int32_t>(y)) % 256;
    //uint8_t const value = (reinterpret_cast<intptr_t>(this)) % 256;
    uint8_t r, g, b;
    get_cmap_val(colormap, value, &r, &g, &b);

    // Draw the object main body
    float const angle = get_angle();
    geom->render(renderer, world_id, pos.x, pos.y, angle,
            SCALE_0_25_TO_0_255(r),
            SCALE_0_25_TO_0_255(g),
            SCALE_0_25_TO_0_255(b),
            255);
}

void PassiveObject::parse_configuration(Configuration const& config, Simulation* simulation) {
    PhysicalObject::parse_configuration(config, simulation);
    colormap = config["colormap"].get(std::string("rainbow"));
}



/************* Factories *************/ // {{{1


Object* object_factory(Simulation* simulation, uint16_t id, float x, float y, b2WorldId world_id, Configuration const& config, LightLevelMap* light_map, size_t userdatasize, std::string const& category) {
    std::string const type = to_lowercase(config["type"].get(std::string("unknown")));
    Object* res = nullptr;

    if (type == "static_light") {
        res = new StaticLightObject(simulation, x, y, light_map, config, category);

    } else if (type == "rotating_ray_of_light") {
        res = new RotatingRayOfLightObject(simulation, x, y, light_map, config, category);

    } else if (type == "alternating_rays_of_light") {
        res = new AlternatingDualRayOfLightObject(simulation, x, y, light_map, config, category);

    } else if (type == "passive_object") {
        res = new PassiveObject(simulation, id, x, y, config, category);

    } else if (type == "pogobot") {
        res = new PogobotObject(simulation, id, x, y, userdatasize, config, category);

    } else if (type == "pogobject") {
        res = new PogobjectObject(simulation, id, x, y, userdatasize, config, category);

    } else if (type == "pogowall") {
        if (simulation->get_boundary_condition() == boundary_condition_t::periodic) {
            // Disable pogowall creation with periodic BC
            res = nullptr;
        } else {
            res = new Pogowall(simulation, id, x, y, userdatasize, config, category);
        }

    } else if (type == "membrane") {
        res = new MembraneObject(simulation, id, x, y, userdatasize, config, category);

    } else if (type == "rectmembrane") {
        res = new RectMembraneObject(simulation, id, x, y, userdatasize, config, category);

    } else if (type == "active_object") {
        res = new ActiveObject(simulation, id, x, y, userdatasize, config, category);

    } else {
        throw std::runtime_error("Unknown object type '" + type + "'.");
    }

    try {
        res->init(world_id);
    } catch (...) {
        delete res;
        glogger->error("Error while creating object type '{}'.", type);
        throw std::runtime_error("Error while creating object type '" + type + "'.");
    }

    return res;
}


void get_cmap_val(std::string const name, uint8_t const value, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (name == "rainbow") {
        rainbow_colormap(value, r, g, b);
    } else if (name == "qualitative") {
        qualitative_colormap(value, r, g, b);
    } else {
        throw std::runtime_error("Unknown colormap '" + name + "'.");
    }
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
