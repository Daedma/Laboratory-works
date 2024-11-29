#pragma once
#include <random>
#include <set>
#include <vector>
#include <cstdint>


class QueueingModel
{
public:

	enum class LineState : uint8_t
	{
		BUSY, AVAILABLE, SCHEDULED
	};

	struct Event
	{
		enum class Types : uint16_t
		{
			ARRIVAL, START, END
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

	void setSimulationTime(double simulationTime);

	void setLambda(double lambda);

	void setBeta(double beta);

	void setNumLines(size_t numLines);

	void setBufferCapacity(size_t bufferCapacity);

	double getSimulationTime() const;

	double getLambda() const;

	double getBeta() const;

	size_t getNumLines() const;

	size_t getBufferCapacity() const;

	size_t getTotalCalls() const;

	size_t getMaxBusyLines() const;

	size_t getCurrentBufferUsage() const;

	size_t getRejectedCalls() const;

	double getEfficiency() const;

	void nextStep();

	void runSimulation();

private:
	double simulationTime;
	double lambda;
	std::gamma_distribution<float> arrivals;
	double beta;
	std::gamma_distribution<float> serviceTime;
	size_t numLines;
	size_t bufferCapacity;

	size_t totalCalls;
	size_t maxBusyLines;
	size_t currentBufferUsage;
	size_t rejectedCalls;
	double efficiency;

	std::multiset<Event> events;

	std::multiset<Event>::iterator currentEvent;

	std::vector<LineState> lines;

	std::mt19937 generator;

	void processArrival();

	void processStart();

	void processEnd();

	float getServiceTime() noexcept
	{

	}

	float generateArrivals();

	void simulate();
};