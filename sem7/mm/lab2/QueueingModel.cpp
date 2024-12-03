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

	std::bernoulli_distribution::param_type failureParams(failureChance);
	failureGenerator.param(failureParams);

	std::exponential_distribution<float>::param_type recoveryTimeParams(recoveryRate);
	recoveryTimeGenerator.param(recoveryTimeParams);

	totalArrivals = 0;

	numBusyLines = 0;

	currentBufferUsage = 0;

	rejectedCalls = 0;

	totalFailures = 0;

	numDisableLines = 0;

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
			case Event::Types::FAILURE:
				processFailureEvent();
				break;
			case Event::Types::SERVICE_START:
				processServiceStartEvent();
				break;
			case Event::Types::SERVICE_END:
				processServiceEndEvent();
				break;
			case Event::Types::RECOVERY_START:
				processRecoveryStartEvent();
				break;
			case Event::Types::RECOVERY_END:
				processRecoveryEndEvent();
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
			events.emplace_hint(pos, currentEvent->timeStamp, Event::Types::SERVICE_START, i);
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

void QueueingModel::processFailureEvent()
{
	++totalFailures;
	Event event = *currentEvent;
	lines[event.line] = LineState::SCHEDULED;
	event.type = Event::Types::RECOVERY_START;
	auto pos = currentEvent;
	++pos;
	events.emplace_hint(pos, event);
}

void QueueingModel::processServiceStartEvent()
{
	++numBusyLines;
	Event event = *currentEvent;
	lines[event.line] = LineState::BUSY;
	event.type = Event::Types::SERVICE_END;
	event.timeStamp = event.timeStamp + getServiceTime();
	events.emplace(event);
}

void QueueingModel::processServiceEndEvent()
{
	--numBusyLines;
	lines[currentEvent->line] = LineState::AVAILABLE;
	bool fail = isFailure();
	if (fail || currentBufferUsage)
	{
		Event event = *currentEvent;
		lines[event.line] = LineState::SCHEDULED;
		if (fail)
		{
			event.type = Event::Types::FAILURE;
		}
		else if (currentBufferUsage)
		{
			event.type = Event::Types::SERVICE_START;
			--currentBufferUsage;
		}
		auto pos = currentEvent;
		++pos;
		events.emplace_hint(pos, event);
	}
}

void QueueingModel::processRecoveryStartEvent()
{
	++numDisableLines;
	Event event = *currentEvent;
	lines[event.line] = LineState::DISABLED;
	event.type = Event::Types::RECOVERY_END;
	event.timeStamp = event.timeStamp + getRecoveryTime();
	events.emplace(event);
}

void QueueingModel::processRecoveryEndEvent()
{
	--numDisableLines;
	lines[currentEvent->line] = LineState::AVAILABLE;
	if (currentBufferUsage)
	{
		Event event = *currentEvent;
		event.type = Event::Types::SERVICE_START;
		--currentBufferUsage;
		auto pos = currentEvent;
		++pos;
		events.emplace_hint(pos, event);
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
	if (std::isnan(failureChance) || failureChance < 0 || failureChance > 1)
	{
		throw std::invalid_argument("Chance of failure must be in range [0, 1]");
	}
	if (!std::isnormal(recoveryRate) || recoveryRate <= 0)
	{
		throw std::invalid_argument("Beta must be positive and normal");
	}
	if (numLines == 0)
	{
		throw std::invalid_argument("Number of lines must be greater than zero");
	}
}
