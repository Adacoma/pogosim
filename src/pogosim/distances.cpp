#include "distances.h"

#include <box2d/box2d.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <set>
#include <iostream>


float euclidean_distance(const b2Vec2& a, const b2Vec2& b) {
    float const dx = a.x - b.x;
    float const dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}


using angles::Interval;
void angles::add_interval(float a, float b, std::vector<Interval>& ivs) {
    std::size_t p = 0;
    while (p < ivs.size() && ivs[p].b < a) ++p;
    while (p < ivs.size() && ivs[p].a <= b) {
        a = std::min(a, ivs[p].a);
        b = std::max(b, ivs[p].b);
        ivs.erase(ivs.begin() + p);
    }
    ivs.insert(ivs.begin() + p, Interval{a,b});
}

bool angles::fully_covered(float a, float b, const std::vector<Interval>& ivs) {
    for (auto const& iv : ivs) {
        if (iv.b <= a) continue;
        if (iv.a >  a) return false;
        a = iv.b;
        if (a >= b)    return true;
    }
    return false;
}

std::unordered_map<GridCell,std::vector<std::size_t>,GridCellHash>
build_spatial_hash(span_t<float> xs,
                   span_t<float> ys,
                   float cell_size) {
    std::unordered_map<GridCell,std::vector<std::size_t>,GridCellHash> h;
    h.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        h[get_grid_cell(xs[i], ys[i], cell_size)].push_back(i);
    return h;
}

std::vector<Candidate>
collect_candidates(std::size_t         i,
                   span_t<float>       xs,
                   span_t<float>       ys,
                   span_t<float>       cx,
                   span_t<float>       cy,
                   span_t<float>       body_rad,
                   span_t<float>       comm_rad,
                   span_t<float>       led_dir,
                   const std::unordered_map<GridCell,
                                            std::vector<std::size_t>,
                                            GridCellHash>& hash,
                   float cell_size,
                   bool  clip_fov) {
    constexpr float k_led_half_fov = M_PI / 2;
    //constexpr float k_led_half_fov = 2* M_PI / 3;
    const GridCell c0 = get_grid_cell(xs[i], ys[i], cell_size);

    std::vector<Candidate> out;
    out.reserve(16);

    for (auto off : precomputed_neighbor_cells) {
        auto it = hash.find({c0.x + off.x, c0.y + off.y});
        if (it == hash.end()) continue;

        for (std::size_t j : it->second) {
            if (j == i) continue;
            float    comm_sq = (comm_rad[i] + body_rad[j]) * (comm_rad[i] + body_rad[j]);

            float dx = cx[j] - xs[i];
            float dy = cy[j] - ys[i];
            float bearing = std::atan2(dy, dx);
            if (clip_fov && !angles::in_fov(bearing, led_dir[i], k_led_half_fov))
                continue;                                               /* self-block */
            float d2 = dx*dx + dy*dy;
            glogger->debug("d2={} comm_sq={} OK:{}", d2, comm_sq, (d2 <= comm_sq));
            if (d2 > comm_sq) continue;

            float d   = std::sqrt(d2);
            float hap = std::asin(std::clamp(body_rad[j] / d, 0.0f, 1.0f));
            out.push_back({j, d2, bearing, hap});
        }
    }
    std::ranges::sort(out, [](auto const& a, auto const& b){ return a.dist_sq < b.dist_sq; });
    return out;
}

std::vector<std::size_t>
filter_visible(const std::vector<Candidate>& cand) {
    using angles::wrap;
    std::vector<Interval> shadow;
    std::vector<std::size_t> visible;
    shadow.reserve(cand.size());

    for (auto const& c : cand) {
        float a = wrap(c.angle - c.half_ap);
        float b = wrap(c.angle + c.half_ap);

        std::vector<std::pair<float,float>> parts;
        if (a <= b) parts.emplace_back(a, b);
        else {
            parts.emplace_back(a,  M_PI);
            parts.emplace_back(-M_PI, b);
        }

        bool vis = false;
        for (auto [s,e] : parts) {
            if (!angles::fully_covered(s, e, shadow)) {
                vis = true;
                angles::add_interval(s, e, shadow);
            }
        }
        if (vis) visible.push_back(c.idx);
    }
    return visible;
}


void find_neighbors(ir_direction dir,
                    std::vector<std::shared_ptr<PogobotObject>>& robots,
                    float max_distance,
                    bool enable_occlusion) {
    //glogger->debug("find_neighbors: dir={}  max_distance={}  enable_occlusion={}", (size_t)dir, max_distance, enable_occlusion);
    /* --- SoA caches ----------------------------------------------------- */
    const std::size_t N = robots.size();
    std::vector<float> led_dir(N);
    std::vector<float> xs(N), ys(N), cx(N), cy(N),
                       body_rad(N), comm_rad(N);

    for (std::size_t i = 0; i < N; ++i) {
        b2Vec2 em = robots[i]->get_IR_emitter_position(dir);
        b2Vec2 ct = robots[i]->get_position();
        xs[i]=em.x; ys[i]=em.y;
        cx[i]=ct.x; cy[i]=ct.y;
        body_rad[i] = robots[i]->radius / VISUALIZATION_SCALE;
        comm_rad[i] = robots[i]->communication_radius / VISUALIZATION_SCALE;
        led_dir[i]  = robots[i]->get_IR_emitter_angle(dir);   /* radians */
    }

    /* --- spatial hash --------------------------------------------------- */
    auto hash = build_spatial_hash(xs, ys, max_distance);

    /* --- per-robot neighbour search ------------------------------------- */
    for (std::size_t i = 0; i < N; ++i) {
        robots[i]->neighbors[dir].clear();

        if (comm_rad[i] == 0.0f)              /* silent robot               */
            continue;

        auto cand = collect_candidates(i, xs, ys, cx, cy,
                                       body_rad, comm_rad,
                                       led_dir,
                                       hash, max_distance,
                                       enable_occlusion);

        std::vector<std::size_t> final_idxs;
        if (enable_occlusion) {
            final_idxs = filter_visible(cand);        /* LOS filter        */
        } else {
            final_idxs.reserve(cand.size());          /* range-only filter */
            for (auto const& c : cand) final_idxs.push_back(c.idx);
        }

        for (std::size_t j : final_idxs)
            robots[i]->neighbors[dir].push_back(robots[j].get());
    }
}


namespace {

/* squared length of a vector */
[[nodiscard]] inline float len_sq(b2Vec2 v) noexcept { return v.x * v.x + v.y * v.y; }

/**
 * @brief Shortest distance P — segment AB.
 *
 * Identical math to what we used in ArenaGeometry; kept here in an unnamed
 * namespace so the compiler can inline it.
 */
[[nodiscard]] float distance_point_segment(b2Vec2 p, b2Vec2 a, b2Vec2 b) noexcept {
    const b2Vec2 ab{b.x - a.x, b.y - a.y};
    const float  ab_len2 = len_sq(ab);
    if (ab_len2 == 0.0f) {          // degenerate edge
        return std::sqrt(len_sq({p.x - a.x, p.y - a.y}));
    }

    const b2Vec2 ap{p.x - a.x, p.y - a.y};
    float t = (ap.x * ab.x + ap.y * ab.y) / ab_len2;   // scalar projection
    t = std::clamp(t, 0.0f, 1.0f);

    const b2Vec2 closest{a.x + t * ab.x, a.y + t * ab.y};
    return std::sqrt(len_sq({p.x - closest.x, p.y - closest.y}));
}

} // namespace

std::vector<float>
compute_wall_distances(ir_direction                           dir,
                       const std::vector<std::shared_ptr<PogobotObject>>& robots,
                       const arena_polygons_t&                arena_polygons) {
    const std::size_t N = robots.size();
    std::vector<float> distances;
    distances.reserve(N);

    /* 1)  Structure‑of‑arrays cache‑friendly layout for robot positions          */
    std::vector<float> xs(N), ys(N);
    for (std::size_t i = 0; i < N; ++i) {
        const b2Vec2 pos = robots[i]->get_IR_emitter_position(dir);
        xs[i] = pos.x;
        ys[i] = pos.y;
    }

    /* 2)  Walk every robot × every arena edge  (usually still very small)        */
    for (std::size_t i = 0; i < N; ++i) {
        const b2Vec2 p{xs[i] * VISUALIZATION_SCALE, ys[i] * VISUALIZATION_SCALE};
        float best = std::numeric_limits<float>::infinity();

        for (const auto& poly : arena_polygons) {
            const std::size_t m = poly.size();
            if (m < 2) continue;

            for (std::size_t k = 0, j = m - 1; k < m; j = k++) {
                best = std::min(best, distance_point_segment(p, poly[j], poly[k]));
                //glogger->info("compute_wall_distances p=({},{}) poly[i]=({},{}), poly[k]=({},{})", p.x, p.y, poly[i].x, poly[i].y, poly[k].x, poly[k].y);
            }
        }
        distances.push_back(best);
    }
    return distances;
}

void find_neighbors_to_pogowalls(std::vector<std::shared_ptr<Pogowall>>& pogowalls, ir_direction dir, std::vector<std::shared_ptr<PogobotObject>>& robots) {
    size_t N = robots.size();

    for (auto wall : pogowalls) {
        arena_polygons_t contours = wall->generate_contours();

        auto dists = compute_wall_distances(dir, robots, contours);
        for (size_t i = 0; i < N; i++) {
            //glogger->info("Not close to wall {}: {} ({})", i, dists[i], robots[i]->communication_radius);
            if (dists[i] <= robots[i]->communication_radius) {
                //glogger->info("Close to wall {}: {} ({})", i, dists[i], robots[i]->communication_radius);
                robots[i]->neighbors[dir].push_back(wall.get());
            }
            if (dists[i] <= wall->communication_radius) {
                wall->neighbors[0].push_back(robots[i].get());
            }
        }
    }
}


/************************* Period boundary conditions *************************/

std::unordered_map<GridCell,std::vector<std::size_t>,GridCellHash>
build_spatial_hash_periodic(span_t<float> xs,
                            span_t<float> ys,
                            float cell_size,
                            b2Vec2 domain_min,
                            float domain_w,
                            float domain_h) {
    std::unordered_map<GridCell,std::vector<std::size_t>,GridCellHash> h;
    h.reserve(xs.size() * 2);

    auto cell = [&](float x, float y){
        return get_grid_cell(x, y, cell_size);
    };

    const float x_min = domain_min.x;
    const float y_min = domain_min.y;
    const float x_max = domain_min.x + domain_w;
    const float y_max = domain_min.y + domain_h;

    for (std::size_t i = 0; i < xs.size(); ++i) {
        float x = xs[i], y = ys[i];

        // Primary insert (real cell)
        GridCell c0 = cell(x, y);
        h[c0].push_back(i);

        // If near a vertical edge, also insert into the wrapped neighbor cell.
        bool near_left  = (x - x_min) < cell_size;
        bool near_right = (x_max - x) < cell_size;
        bool near_bot   = (y - y_min) < cell_size;
        bool near_top   = (y_max - y) < cell_size;

        // Horizontal ghosts
        if (near_left) {
            GridCell cw = cell(x + domain_w, y);
            h[cw].push_back(i);
        }
        if (near_right) {
            GridCell ce = cell(x - domain_w, y);
            h[ce].push_back(i);
        }

        // Vertical ghosts
        if (near_bot) {
            GridCell cs = cell(x, y + domain_h);
            h[cs].push_back(i);
        }
        if (near_top) {
            GridCell cn = cell(x, y - domain_h);
            h[cn].push_back(i);
        }

        // Corner ghosts (up to 4)
        if (near_left && near_bot) {
            h[cell(x + domain_w, y + domain_h)].push_back(i);
        }
        if (near_left && near_top) {
            h[cell(x + domain_w, y - domain_h)].push_back(i);
        }
        if (near_right && near_bot) {
            h[cell(x - domain_w, y + domain_h)].push_back(i);
        }
        if (near_right && near_top) {
            h[cell(x - domain_w, y - domain_h)].push_back(i);
        }
    }
    return h;
}

std::vector<Candidate>
collect_candidates_periodic(std::size_t   i,
                            span_t<float> xs,
                            span_t<float> ys,
                            span_t<float> cx,
                            span_t<float> cy,
                            span_t<float> body_rad,
                            span_t<float> comm_rad,
                            span_t<float> led_dir,
                            const std::unordered_map<GridCell,
                                                     std::vector<std::size_t>,
                                                     GridCellHash>& hash,
                            float cell_size,
                            bool  clip_fov,
                            b2Vec2 domain_min,
                            float domain_w,
                            float domain_h) {
    constexpr float k_led_half_fov = M_PI / 2;
    const GridCell c0 = get_grid_cell(xs[i], ys[i], cell_size);

    std::vector<Candidate> out;
    out.reserve(16);

    // Emitter position and robot center of i in Box2D units
    const b2Vec2 emi_i{xs[i], ys[i]};

    for (auto off : precomputed_neighbor_cells) {
        auto it = hash.find({c0.x + off.x, c0.y + off.y});
        if (it == hash.end()) continue;

        for (std::size_t j : it->second) {
            if (j == i) continue;

            float comm = (comm_rad[i] + body_rad[j]);
            float comm_sq = comm * comm;

            const b2Vec2 ctr_j{cx[j], cy[j]};

            // Toroidal delta from emitter(i) to center(j)
            b2Vec2 d = torus_delta(emi_i, ctr_j, domain_min, domain_w, domain_h);

            float bearing = std::atan2(d.y, d.x);
            if (clip_fov && !angles::in_fov(bearing, led_dir[i], k_led_half_fov))
                continue;

            float d2 = d.x * d.x + d.y * d.y;
            if (d2 > comm_sq) continue;

            float dlen = std::sqrt(d2);
            float hap  = std::asin(std::clamp(body_rad[j] / dlen, 0.0f, 1.0f));
            out.push_back({j, d2, bearing, hap});
        }
    }
    std::ranges::sort(out, [](auto const& a, auto const& b){ return a.dist_sq < b.dist_sq; });
    return out;
}

void find_neighbors_periodic(
    ir_direction dir,
    std::vector<std::shared_ptr<PogobotObject>>& robots,
    float max_distance,
    b2Vec2 domain_min,
    float domain_w,
    float domain_h,
    bool enable_occlusion) {

    const std::size_t N = robots.size();
    std::vector<float> led_dir(N);
    std::vector<float> xs(N), ys(N), cx(N), cy(N),
                       body_rad(N), comm_rad(N);

    for (std::size_t i = 0; i < N; ++i) {
        b2Vec2 em = robots[i]->get_IR_emitter_position(dir);
        b2Vec2 ct = robots[i]->get_position();
        xs[i]=em.x; ys[i]=em.y;
        cx[i]=ct.x; cy[i]=ct.y;
        body_rad[i] = robots[i]->radius / VISUALIZATION_SCALE;
        comm_rad[i] = robots[i]->communication_radius / VISUALIZATION_SCALE;
        led_dir[i]  = robots[i]->get_IR_emitter_angle(dir);
    }

    auto hash = build_spatial_hash_periodic(xs, ys, max_distance,
                                            domain_min, domain_w, domain_h);

    for (std::size_t i = 0; i < N; ++i) {
        robots[i]->neighbors[dir].clear();
        if (comm_rad[i] == 0.0f) continue;

        auto cand = collect_candidates_periodic(i, xs, ys, cx, cy,
                                                body_rad, comm_rad, led_dir,
                                                hash, max_distance,
                                                enable_occlusion,
                                                domain_min, domain_w, domain_h);

        std::vector<std::size_t> final_idxs;
        if (enable_occlusion) {
            final_idxs = filter_visible(cand);
        } else {
            final_idxs.reserve(cand.size());
            for (auto const& c : cand) final_idxs.push_back(c.idx);
        }

        for (std::size_t j : final_idxs)
            robots[i]->neighbors[dir].push_back(robots[j].get());
    }
}



// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
