#pragma once

#include <stdexcept>

#include "shape.hpp"

class ellipse : public shape
{
    double m_x_radius;
    double m_y_radius;
    double m_z_radius;
public:
    ellipse(double x_radius_, double y_radius_, double z_radius_)
    {
        x_radius(x_radius_);
        y_radius(y_radius_);
        z_radius(z_radius_);
    }

    void x_radius(double new_x_radius_)
    {
        if (new_x_radius_ <= 0.)
        {
            throw std::invalid_argument{ "x radius must be positive" };
        }
        m_x_radius = new_x_radius_;
    }

    double x_radius() const noexcept
    {
        return m_x_radius;
    }

    void y_radius(double new_y_radius_)
    {
        if (new_y_radius_ <= 0.)
        {
            throw std::invalid_argument{ "y radius must be positive" };
        }
        m_y_radius = new_y_radius_;
    }

    double y_radius() const noexcept
    {
        return m_y_radius;
    }

    void z_radius(double new_z_radius_)
    {
        if (new_z_radius_ <= 0.)
        {
            throw std::invalid_argument{ "z radius must be positive" };
        }
        m_z_radius = new_z_radius_;
    }

    double z_radius() const noexcept
    {
        return m_z_radius;
    }

    vec_t normal(const vec_t& point) const override;

	void draw(LibBoard::Board& board, const LibBoard::Color &color) const override;

protected:
    std::vector<vec_t::value_type> intersection_points_Impl(const ray& ray_) const override;
};