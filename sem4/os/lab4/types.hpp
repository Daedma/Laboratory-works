#pragma once

#define SEMAPHORE_NAME L"LocalMaxClientProcesses_Semaphore"

#define OSLAB_PIPENAME R"(\\.\pipe\OSLABPipe)"

struct message
{
	float lifetime;
	size_t number;
};
