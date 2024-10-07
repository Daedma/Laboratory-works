#include "PlanetSystem.hpp"

namespace
{
	constexpr double G = 6.6743e-11;
}

void PlanetSystem::makeEulerStep(size_t step)
{
	size_t planetsCount = planets.size();
	for (size_t i = 0; i != planetsCount; ++i)
	{
		Point an = calcAcceleration(i, step);
		velocities[i][step + 1] = velocities[i][step] + an * timeStep;
		coordinates[i][step + 1] = coordinates[i][step] + velocities[i][step] * timeStep;
	}
}

void PlanetSystem::makeEulerCromerStep(size_t step)
{
	size_t planetsCount = coordinates.size();
	for (size_t i = 0; i != planetsCount; ++i)
	{
		Point an = calcAcceleration(i, step);
		velocities[i][step + 1] = velocities[i][step] + an * timeStep;
		coordinates[i][step + 1] = coordinates[i][step] + velocities[i][step + 1] * timeStep;
	}
}

void PlanetSystem::makeVerletStep(size_t step)
{
	size_t planetsCount = planets.size();
	for (size_t i = 0; i != planetsCount; ++i)
	{
		Point an = calcAcceleration(i, step);
		Point prevVelocity = step > 0 ? (coordinates[i][step] - coordinates[i][step - 1]) / timeStep :
			velocities[i][step] + an * timeStep * 0.5;
		Point nextVelocity = prevVelocity + an * timeStep;
		Point nextCoordinate = coordinates[i][step] + nextVelocity * timeStep;
		velocities[i][step + 1] = nextVelocity;
		coordinates[i][step + 1] = nextCoordinate;
	}
}

void PlanetSystem::makeBeemanStep(size_t step)
{
	makeEulerStep(step);
	if (step != 0)
	{
		size_t planetsCount = planets.size();
		for (size_t i = 0; i != planetsCount; ++i)
		{
			Point an = calcAcceleration(i, step);
			Point anm1 = calcAcceleration(i, step - 1);
			Point anp1 = calcAcceleration(i, step + 1);
			coordinates[i][step + 1] = coordinates[i][step] + velocities[i][step] * timeStep +
				(4 * an - anm1) / 6 * timeStep * timeStep;
			velocities[i][step + 1] = velocities[i][step] +
				(2 * anp1 + 5 * an - anm1) / 6 * timeStep;
		}
	}
}

double PlanetSystem::getEnergyValue() const
{
	size_t lastStep = getLastStep();
	double totalEnergy = 0.0;
	size_t planetsCount = planets.size();

	// вычисление кинетической энергии
	for (size_t i = 0; i != planetsCount; ++i)
	{
		double kineticEnergy = 0.5 * planets[i].mass * boost::qvm::mag_sqr(velocities[i][lastStep]);
		totalEnergy += kineticEnergy;
	}
	// вычисление потенциальной энергии
	for (size_t i = 0; i != planetsCount; ++i)
	{
		Point icoords = coordinates[i][lastStep];
		double imass = planets[i].mass;
		for (size_t j = i + 1; j != planetsCount; ++j)
		{
			double r = boost::qvm::mag(icoords - coordinates[j][lastStep]);
			if (r > 0)
			{
				double potentialEnergy = -G * imass * planets[j].mass / r;
				totalEnergy += potentialEnergy;
			}
		}
	}

	return totalEnergy;
}

PlanetSystem::Point PlanetSystem::getVelocityValue() const
{
	size_t lastStep = getLastStep();
	Point totalVelocity{ 0., 0. };
	double totalMass = 0.0;
	size_t planetsCount = planets.size();
	for (size_t i = 0; i != planetsCount; ++i)
	{
		totalVelocity += planets[i].mass * velocities[i][lastStep];
		totalMass += planets[i].mass;
	}
	if (totalMass > 0)
	{
		totalVelocity /= totalMass;
	}
	return totalVelocity;
}

PlanetSystem::Point PlanetSystem::calcAcceleration(size_t nplanet, size_t step) const
{
	Point acceleration{ 0., 0. };
	size_t planetsCount = planets.size();
	for (size_t i = 0; i != planetsCount; ++i)
	{
		if (i != nplanet)
		{
			Point distanceVector = planets[i].pos - planets[nplanet].pos;
			double sqrDistance = boost::qvm::mag_sqr(distanceVector);
			if (sqrDistance > 0)
			{ // Avoid division by zero
				Point direction = distanceVector / std::sqrt(sqrDistance);
				acceleration += G * planets[i].mass / sqrDistance * direction;
			}
		}
	}
	return acceleration;
}

void PlanetSystem::allocMemory()
{
	size_t stepsCount = std::ceil(simulationTime / timeStep);
	coordinates.resize(planets.size());
	for (auto& i : coordinates)
	{
		i.resize(stepsCount);
	}
	velocities.resize(planets.size());
	for (auto& i : velocities)
	{
		i.resize(stepsCount);
	}
}

void PlanetSystem::setInitialConditions()
{
	for (size_t i = 0; i != planets.size(); ++i)
	{
		coordinates[i][0] = planets[i].pos;
		velocities[i][0] = planets[i].velocity;
	}
}

void PlanetSystem::run()
{
	isValidSetup();
	allocMemory();
	setInitialConditions();
	workThread.reset(new std::thread{
		[this]() {
			size_t stepsCount = std::ceil(simulationTime / timeStep);
			size_t curStep = 0;
			this->inProgress = true;
			while (curStep != stepsCount)
			{
				(this->*stepper)(curStep);
				++curStep;
				this->previousStep = curStep - 1;
			}
			this->inProgress = false;
		}
		});
}

void PlanetSystem::stop()
{
	workThread->detach();
	workThread.reset();
	inProgress = false;
}