#pragma once

#define SEMAPHORE_NAME L"LocalMaxClientProcesses_Semaphore"

struct message
{
	float lifetime;
	size_t number;
};
