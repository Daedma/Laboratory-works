#include "QueueingModel.hpp"

void QueueingModel::nextStep()
{
	++currentEvent;
	if (currentEvent != events.end())
	{
		switch (currentEvent->type)
		{
		case Event::Types::ARRIVAL:
			processArrival();
			break;
		case Event::Types::START:
			processStart();
			break;
		case Event::Types::END:
			processEnd();
			break;
		default:
			break;
		}
	}
}

void QueueingModel::processArrival()
{
	++totalCalls;
	for (uint32_t i = 0; i != lines.size(); ++i)
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

void QueueingModel::processStart()
{
	++maxBusyLines;
	Event event = *currentEvent;
	lines[event.line] = LineState::BUSY;
	event.type = Event::Types::END;
	event.timeStamp = event.timeStamp + serviceTime(generator);
	events.emplace(event);
}

void QueueingModel::processEnd()
{
	--maxBusyLines;
	lines[currentEvent->line] = LineState::AVAILABLE;
	if (currentBufferUsage)
	{
		auto pos = currentEvent;
		++pos;
		events.emplace_hint(pos, currentEvent->timeStamp, Event::Types::ARRIVAL);
		--currentBufferUsage;
	}
}