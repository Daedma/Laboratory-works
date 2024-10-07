#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include <stdexcept>
#include "qvm_lite.hpp"

class PlanetSystem
{
public:
	using Point = boost::qvm::vec<double, 2>;

	enum DiffSchemes
	{
		EULER, VERLET, EULER_CROMER, BEEMAN
	};

	struct PlanetParams
	{
		Point pos;
		Point velocity;
		double mass;
	};


	void setDiffScheme(DiffSchemes type)
	{
		throwExceptionIfInProgress();
		switch (type)
		{
		case DiffSchemes::EULER:
			stepper = &PlanetSystem::makeEulerStep;
			break;
		case DiffSchemes::VERLET:
			stepper = &PlanetSystem::makeVerletStep;
			break;
		case DiffSchemes::EULER_CROMER:
			stepper = &PlanetSystem::makeEulerCromerStep;
			break;
		case DiffSchemes::BEEMAN:
			stepper = &PlanetSystem::makeBeemanStep;
			break;
		default:
			break;
		}
	}

	void setTimeStep(double step)
	{
		throwExceptionIfInProgress();
		timeStep = step;
	}

	void setSimulationTime(double time)
	{
		throwExceptionIfInProgress();
		simulationTime = time;
	}

	void setSystemParams(const std::vector<PlanetParams>& params)
	{
		planets = params;
	}

	void run();

	void stop();

	size_t getLastStep() const
	{
		return previousStep.load();
	}

	const std::vector<std::vector<Point>>& getPaths() const
	{
		return coordinates;
	}

	double getEnergyValue() const;

	Point getVelocityValue() const;

private:
	// Concurency
	std::atomic_bool inProgress = false;
	std::atomic_size_t previousStep = -1;
	std::unique_ptr<std::thread> workThread;

	// Initial parameters
	double timeStep = 0.;
	void(PlanetSystem::* stepper)(size_t) = nullptr;
	double simulationTime = 0.;
	std::vector<PlanetParams> planets;

	// Result
	std::vector<std::vector<Point>> coordinates;
	std::vector<std::vector<Point>> velocities;

private:
	Point calcAcceleration(size_t nplanet, size_t step) const;

	void makeEulerStep(size_t step);

	void makeVerletStep(size_t step);

	void makeEulerCromerStep(size_t step);

	void makeBeemanStep(size_t step);

	void allocMemory();

	void setInitialConditions();

	void isValidSetup() const
	{
		if (stepper == nullptr)
		{
			throw std::runtime_error("Differential scheme is not set.");
		}
		if (timeStep <= 0)
		{
			throw std::runtime_error("Time step must be positive.");
		}
		if (simulationTime <= 0)
		{
			throw std::runtime_error("Simulation time must be positive.");
		}
		if (planets.empty())
		{
			throw std::runtime_error("No planets in the system.");
		}

		// Check if all planets have valid parameters
		for (const auto& planet : planets)
		{
			using namespace boost::qvm;
			// Check if the mass is positive
			if (planet.mass <= 0)
			{
				throw std::runtime_error("Planet mass must be positive.");
			}

			// Check if the position and velocity are valid (not NaN or infinite)
			if (!std::isfinite(X(planet.pos)) || !std::isfinite(Y(planet.pos)) ||
				!std::isfinite(X(planet.velocity)) || !std::isfinite(Y(planet.velocity)))
			{
				throw std::runtime_error("Planet mass must be finite.");
			}
		}
	}

	void throwExceptionIfInProgress() const
	{
		if (inProgress)
		{
			throw std::runtime_error{ "Attempt to change attribute values ​​during modeling process!" };
		}
	}
};