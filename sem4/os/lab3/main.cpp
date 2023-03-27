#include <unistd.h>
#include <iostream>
#include <string>
#include <cstring>
#include <signal.h>

void print_info(const char* progname)
{
	std::cout <<
		progname << " - calculates the sum of a series with a given precision.\n"
		"Usage: " << progname << " [<input_file=input.txt> [<output_file=output.txt>]].\n"
		"In <input_file> - X and precision.\n"
		"In <output_file> - sum and precision.\n";
}

int main(int argc, char const* argv[])
{
	if (argc > 1 && strcmp(argv[1], "--help") == 0)
	{
		print_info(argv[0]);
		return EXIT_SUCCESS;
	}
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
		// fclose(stdout);
		close(pipefd1[1]);
		dup2(pipefd1[0], STDIN_FILENO);
		close(pipefd1[0]);
		// fclose(stdin);
		execl("client", std::to_string(c_pid).c_str(), NULL);
	}
	else if (c_pid == 0) // Обработка дочернего процесса
	{
		close(pipefd1[0]);
		dup2(pipefd1[1], STDOUT_FILENO);
		// fclose(stdout);
		close(pipefd2[1]);
		dup2(pipefd2[0], STDIN_FILENO);
		close(pipefd2[0]);
		// fclose(stdin);
		execl("server", NULL);
	}
}
