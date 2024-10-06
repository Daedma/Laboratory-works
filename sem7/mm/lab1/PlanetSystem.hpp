#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include "qvm_lite.hpp"

class PlanetSystem
{
public:
	enum DiffSchemes
	{
		EULER, VERLET, EULER_CROMER, BEEMAN
	};

	struct PlanetParams
	{
		double posX;
		double posY;
		double velocityX;
		double velocityY;
		double mass;
	};

	using Point = boost::qvm::vec<double, 2>;

	void setDiffScheme(DiffSchemes type)
	{
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
		timeStep = step;
	}

	void setSimulationTime(double time)
	{
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
	std::atomic_bool inProgress;
	std::atomic_size_t previousStep;
	std::unique_ptr<std::thread> workThread;

	// Initial parameters
	double timeStep;
	void(PlanetSystem::* stepper)(size_t);
	double simulationTime;
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
};