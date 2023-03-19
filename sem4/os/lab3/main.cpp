#include <unistd.h>
#include <iostream>

int main(int argc, char const* argv[])
{
	int pipefd1[2]; // Server to client
	pipe(pipefd1);
	int pipefd2[2]; // Client to server
	pipe(pipefd2);
	pid_t c_pid = fork();
	if (c_pid == -1)
	{
		std::cerr << "Ошибка при создании дочернего процесса\n";
		return EXIT_FAILURE;
	}
	else if (c_pid > 0) // Обработка родительского процесса
	{
		close(pipefd2[0]);
		dup2(pipefd2[1], STDOUT_FILENO);
		close(pipefd1[1]);
		dup2(pipefd1[0], STDIN_FILENO);
		execl("client", NULL);
	}
	else if (c_pid == 0) // Обработка дочернего процесса
	{
		close(pipefd1[0]);
		dup2(pipefd1[1], STDOUT_FILENO);
		close(pipefd2[1]);
		dup2(pipefd2[0], STDIN_FILENO);
		execl("server", NULL);
	}
}
