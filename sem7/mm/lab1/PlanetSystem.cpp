#include "PlanetSystem.hpp"

namespace
{
	constexpr double G = 6.6743e-11;
}

void PlanetSystem::makeEulerStep(size_t step)
{
	size_t planetsCount = coordinates.size();
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
	size_t planetsCount = coordinates.size();
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
		size_t planetsCount = coordinates.size();
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