#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "boost/qvm_lite.hpp"

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

	const std::vector<PlanetParams>& getSystemParams() const
	{
		return planets;
	}

	void makeStep()
	{
		if (inProgress)
		{
			size_t stepsCount = std::ceil(simulationTime / timeStep);
			size_t curStep = previousStep;
			(this->*stepper)(curStep);
			++previousStep;
			if (curStep == stepsCount - 1)
			{
				inProgress = false;
			}
		}
	}

	void run(bool inConcurency = false);

	void stop();

	size_t getLastStep() const
	{
		return previousStep.load();
	}

	double getCurrentTime(size_t step) const
	{
		return timeStep * step;
	}

	const std::vector<std::vector<Point>>& getPaths() const
	{
		return coordinates;
	}

	double getEnergyValue(size_t step) const;

	double getEnergyValue() const
	{
		return getEnergyValue(getLastStep());
	}

	Point getVelocityValue(size_t step) const;

	Point getVelocityValue() const
	{
		return getVelocityValue(getLastStep());
	}

	bool isInProgress() const
	{
		return inProgress;
	}

	void setViscosity(double value)
	{
		throwExceptionIfInProgress();
		viscosity = value;
	}

	static PlanetParams createDefaultPlanet(size_t n);

private:
	// Concurency
	std::atomic_bool inProgress;
	std::atomic_size_t previousStep;

	// Initial parameters
	double timeStep = 0.;
	void(PlanetSystem::* stepper)(size_t) = nullptr;
	double simulationTime = 0.;
	std::vector<PlanetParams> planets;
	double viscosity = 0.;

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
		if (viscosity < 0.)
		{
			throw std::runtime_error("Viscosity must be non-negative.");
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