#include <cmath>

#include "QueueingModel.hpp"

void QueueingModel::startSimulation()
{
	checkIsRunning();
	checkParams();

	lines.clear();
	lines.resize(numLines, LineState::AVAILABLE);

	std::exponential_distribution<float>::param_type arrivalParams(arrivalRate);
	arrivalTimeGenerator.param(arrivalParams);

	std::exponential_distribution<float>::param_type serviceTimeParams(reverseServiceTimeMean);
	serviceTimeGenerator.param(serviceTimeParams);

	totalArrivals = 0;

	numBusyLines = 0;

	currentBufferUsage = 0;

	rejectedCalls = 0;

	events.clear();

	randomGenerator.seed(std::random_device{}());

	generateArrivals();

	currentEvent = events.cbegin();

	isRunning = true;
}

void QueueingModel::generateArrivals()
{
	float time = 0.f;
	std::multiset<Event>::const_iterator last = events.cend();
	while (time <= simulationTime)
	{
		events.emplace_hint(last, time, Event::Types::ARRIVAL);
		time += getNextArrivalTime();
	}
}

void QueueingModel::nextStep()
{
	if (isRunning)
	{
		if (currentEvent != events.end())
		{
			switch (currentEvent->type)
			{
			case Event::Types::ARRIVAL:
				processArrivalEvent();
				break;
			case Event::Types::START:
				processStartEvent();
				break;
			case Event::Types::END:
				processEndEvent();
				break;
			default:
				break;
			}
			++currentEvent;
		}
		else
		{
			isRunning = false;
		}
	}
}

void QueueingModel::processArrivalEvent()
{
	++totalArrivals;
	for (uint16_t i = 0; i != lines.size(); ++i)
	{
		if (lines[i] == LineState::AVAILABLE)
		{
			lines[i] = LineState::SCHEDULED;
			auto pos = currentEvent;
			++pos;
			events.emplace_hint(pos, currentEvent->timeStamp, Event::Types::START, i);
			return;
		}
	}
	if (currentBufferUsage < bufferCapacity)
	{
		++currentBufferUsage;
	}
	else
	{
		++rejectedCalls;
	}
}

void QueueingModel::processStartEvent()
{
	++numBusyLines;
	Event event = *currentEvent;
	lines[event.line] = LineState::BUSY;
	event.type = Event::Types::END;
	event.timeStamp = event.timeStamp + getServiceTime();
	events.emplace(event);
}

void QueueingModel::processEndEvent()
{
	--numBusyLines;
	lines[currentEvent->line] = LineState::AVAILABLE;
	if (currentBufferUsage)
	{
		auto pos = currentEvent;
		++pos;
		events.emplace_hint(pos, currentEvent->timeStamp, Event::Types::ARRIVAL);
		--currentBufferUsage;
	}
}

void QueueingModel::checkParams()
{
	if (!std::isnormal(simulationTime) && simulationTime < 0)
	{
		throw std::invalid_argument("Simulation time must be non-negative and normal");
	}
	if (!std::isnormal(arrivalRate) || arrivalRate <= 0)
	{
		throw std::invalid_argument("Lambda must be positive and normal");
	}
	if (!std::isnormal(reverseServiceTimeMean) || reverseServiceTimeMean <= 0)
	{
		throw std::invalid_argument("Beta must be positive and normal");
	}
	if (numLines == 0)
	{
		throw std::invalid_argument("Number of lines must be greater than zero");
	}
}
