#pragma once
#include <random>
#include <set>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <utility>


class QueueingModel
{
public:

	enum class LineState : uint8_t
	{
		BUSY, AVAILABLE, SCHEDULED, DISABLED
	};

	struct Event
	{
		enum class Types : uint16_t
		{
			ARRIVAL, FAILURE, SERVICE_START, SERVICE_END, RECOVERY_START, RECOVERY_END
		};

		float timeStamp;
		Types type;
		uint16_t line;

		Event(float aTimeStamp, Types aType, uint16_t aLine = static_cast<uint16_t>(-1)) noexcept :
			timeStamp(aTimeStamp), type(aType), line(aLine)
		{}

		bool operator<(const Event& rhs) const noexcept
		{
			return timeStamp < rhs.timeStamp;
		}
	};

	void setSimulationTime(float aSimulationTime)
	{
		checkIsRunning();
		if (aSimulationTime < 0)
		{
			throw std::invalid_argument{ "Simulation time must be non-negative" };
		}
		simulationTime = aSimulationTime;
	}

	void setArrivalRate(float aArrivalRate)
	{
		checkIsRunning();
		if (aArrivalRate < 0)
		{
			throw std::invalid_argument{ "Lambda must be non-negative" };
		}
		arrivalRate = aArrivalRate;
	}

	void setReverseServiceTimeMean(float aReverseServiceTimeMean)
	{
		checkIsRunning();
		if (aReverseServiceTimeMean < 0)
		{
			throw std::invalid_argument{ "Beta must be non-negative" };
		}
		reverseServiceTimeMean = aReverseServiceTimeMean;
	}

	void setFailureChance(float aFailureChance)
	{
		checkIsRunning();
		if (aFailureChance < 0 || aFailureChance > 1)
		{
			throw std::invalid_argument{ "Chance of failure must be in range [0, 1]" };
		}
		failureChance = aFailureChance;
	}

	void setRecoveryRate(float aRecoveryRate)
	{
		checkIsRunning();
		if (aRecoveryRate < 0)
		{
			throw std::invalid_argument{ "Recovery rate must be non-negative" };
		}
		recoveryRate = aRecoveryRate;
	}

	void setNumLines(uint16_t aNumLines)
	{
		checkIsRunning();
		numLines = aNumLines;
	}

	void setBufferCapacity(size_t aBufferCapacity)
	{
		checkIsRunning();
		bufferCapacity = aBufferCapacity;
	}

	float getSimulationTime() const noexcept
	{
		return simulationTime;
	}

	float getArrivalRate() const noexcept
	{
		return arrivalRate;
	}

	float getReverseServiceTimeMean() const noexcept
	{
		return reverseServiceTimeMean;
	}

	float getFailureChance() const noexcept
	{
		return failureChance;
	}

	float getRecoveryRate() const noexcept
	{
		return recoveryRate;
	}

	uint16_t getNumLines() const noexcept
	{
		return numLines;
	}

	size_t getBufferCapacity() const noexcept
	{
		return bufferCapacity;
	}

	size_t getTotalArrivals() const noexcept
	{
		return totalArrivals;
	}

	size_t getTotalFailures() const noexcept
	{
		return totalFailures;
	}

	size_t getNumBusyLines() const noexcept
	{
		return numBusyLines;
	}

	size_t getNumDisableLines() const noexcept
	{
		return numDisableLines;
	}

	size_t getCurrentBufferUsage() const noexcept
	{
		return currentBufferUsage;
	}

	size_t getRejectedCalls() const noexcept
	{
		return rejectedCalls;
	}

	double getEfficiency() const noexcept
	{
		return 1 - (totalArrivals ? static_cast<float>(rejectedCalls) / totalArrivals : 0.);
	}

	std::pair<std::multiset<Event>::const_iterator, std::multiset<Event>::const_iterator>
		getProcesedEvents() const noexcept
	{
		return { events.cbegin(), currentEvent };
	}

	bool getIsRunning() const noexcept
	{
		return isRunning;
	}

	void stopSimulation() noexcept
	{
		isRunning = false;
	}

	void startSimulation();

	void nextStep();

private:
	float simulationTime = NAN;

	float arrivalRate = NAN;

	std::exponential_distribution<float> arrivalTimeGenerator;

	float reverseServiceTimeMean = NAN;

	std::exponential_distribution<float> serviceTimeGenerator;

	float failureChance = 0;

	std::bernoulli_distribution failureGenerator;

	float recoveryRate = NAN;

	std::exponential_distribution<float> recoveryTimeGenerator;

	uint16_t numLines = 0;

	size_t bufferCapacity = 0;

	size_t totalArrivals = 0;

	size_t numBusyLines = 0;

	size_t currentBufferUsage = 0;

	size_t rejectedCalls = 0;

	size_t totalFailures = 0;

	size_t numDisableLines = 0;

	std::multiset<Event> events;

	std::multiset<Event>::const_iterator currentEvent;

	std::vector<LineState> lines;

	std::mt19937 randomGenerator;

	bool isRunning = false;

	void checkIsRunning()
	{
		if (isRunning)
		{
			throw std::runtime_error{ "Simulation is running, parameters cannot be changed until end of simulation." };
		}
	}

	void checkParams();

	void processArrivalEvent();

	void processFailureEvent();

	void processServiceStartEvent();

	void processServiceEndEvent();

	void processRecoveryStartEvent();

	void processRecoveryEndEvent();

	float getServiceTime() noexcept
	{
		return serviceTimeGenerator(randomGenerator);
	}

	float getNextArrivalTime() noexcept
	{
		return arrivalTimeGenerator(randomGenerator);
	}

	bool isFailure() noexcept
	{
		return failureGenerator(randomGenerator);
	}

	float getRecoveryTime() noexcept
	{
		return recoveryTimeGenerator(randomGenerator);
	}

	void generateArrivals();
};