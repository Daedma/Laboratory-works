// UNIX
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Other
#include <sstream>
#include <numeric>
#include <map>
#include <iterator>
#include <iostream>
#include <string>
#include <functional>
#include <string_view>

class file_operations
{
public:
	using command_t = bool(const std::vector<std::string_view>);

	file_operations() = delete;

	static int execute(std::string_view command, const std::vector<std::string_view>& args);

	static int execute(const std::string& args);

	static const std::string& actions();

private:
	static bool copy(const std::vector<std::string_view>& args);

	static bool move(const std::vector<std::string_view>& args);

	static bool info(const std::vector<std::string_view>& args);

	static bool mode(const std::vector<std::string_view>& args);

	static bool help(const std::vector<std::string_view>& args);
private:

	static void init();

private:

	static bool is_initialized;

	const static std::map<std::string_view, std::function<command_t>> commands;
};

bool file_operations::is_initialized = false;

int main(int argc, char const* argv[])
{
	if (argc > 1)
	{
		return	file_operations::execute(argv[1], { argv + 1, argv + argc });
	}
	else
	{
		std::cout << file_operations::actions() << std::endl;
		std::string command;
		do
		{
			std::cout << ": ";
			std::getline(std::cin, command);
		} while (command.empty());
		return file_operations::execute(command);
	}
	return 0;
}

bool file_operations::copy(const std::vector<std::string_view>& args)
{
	static constexpr int BUFFER_SIZE = 256;

	if (args.size() < 2) return false;

	const char* source_path = args[0].data();
	const char* dest_path = args[1].data();

	int source_desc = open(source_path, O_RDONLY);
	if (source_desc == -1) return false;
	int dest_desc = open(dest_path, O_WRONLY);
	if (dest_desc == -1) return false;

	ftruncate(dest_desc, 0);

	char* buffer = new char[BUFFER_SIZE];
	size_t last_read = 0;
	do
	{
		last_read = read(source_desc, buffer, BUFFER_SIZE);
		write(dest_desc, buffer, last_read);
	} while (last_read != BUFFER_SIZE);
	fsync(dest_desc);
	close(source_desc);
	close(dest_desc);
	delete[] buffer;

	return true;
}

bool file_operations::move(const std::vector<std::string_view>& args)
{
	if (args.size() < 2) return false;
	const char* old_path = args[0].data();
	const char* new_path = args[1].data();
	return rename(old_path, new_path) != -1;
}

bool file_operations::info(const std::vector<std::string_view>& args)
{
	if (args.size() < 1) return false;
	const char* file_name = args[0].data();
	bool as_link = args.size() > 1 && args[1] == "-l";
	struct stat file_info;
	int ret = as_link ? lstat(file_name, &file_info) : stat(file_name, &file_info);
	if (ret == -1) return false;
	std::cout << "ID of device containing file: " << file_info.st_dev << '\n'
		<< "Inode number: " << file_info.st_ino << '\n'
		<< "File type and mode: " << file_info.st_mode << '\n'
		<< "Number of hard links: " << file_info.st_nlink << '\n'
		<< "User ID of owner: " << file_info.st_uid << '\n'
		<< "Group ID of owner: " << file_info.st_gid << '\n'
		<< "Device ID (if special file): " << file_info.st_rdev << '\n'
		<< "Total size, in bytes: " << file_info.st_size << '\n'
		<< "Block size for filesystem I/O: " << file_info.st_blksize << '\n'
		<< "Number of 512B blocks allocated: " << file_info.st_blocks << '\n'
		<< "Time of last access (in seconds): " << file_info.st_atim.tv_sec << '\n'
		<< "Time of last modification (in seconds): " << file_info.st_mtim.tv_sec << '\n'
		<< "Time of last status change (in seconds): " << file_info.st_ctim.tv_sec << std::endl;
	return true;if (ret == -1) return false;
}

bool file_operations::mode(const std::vector<std::string_view>& args)
{
	const static std::map<std::string_view, mode_t> flags_map = { { "ISUID", S_ISUID },
		{ "ISGID", S_ISGID }, { "ISVTX", S_ISVTX }, { "IRUSR", S_IRUSR }, { "IWUSR", S_IWUSR }, { "IXUSR", S_IXUSR }, { "IRGRP", S_IRGRP },
		{ "IWGRP", S_IWGRP }, { "IXGRP", S_IXGRP }, { "IROTH", S_IROTH }, { "IWOTH", S_IWOTH }, { "IXOTH", S_IXOTH } };
	if (args.size() < 1) return false;
	const char* file_name = args[0].data();
	mode_t flags = 0;
	for (const auto& i : args)
	{
		try
		{
			flags |= flags_map.at(i);
		}
		catch (...)
		{
			return false;
		}
	}
	int ret = chmod(file_name, flags);
	return ret != -1;
}

bool file_operations::help(const std::vector<std::string_view>& args)
{
	std::cout <<
		"fops                           - file operations.\n"
		"fops [<command>] [<args>...]   - run <command> with <args>.\n"
		"List of available commands:\n"
		<< actions();
	return true;
}

const std::string& file_operations::actions()
{
	static const std::string& ops =
		"copy <source>   <destination>  - copy data from <source> file to <destination>\n"
		"move <old_path> <new_path>     - move file from <old_path> to <new_path>\n"
		"info <file>                    - print information about <file>\n"
		"mode <file>     [<flags>...]   - set mode of <file> with <flags>\n";
	return ops;
}

int file_operations::execute(std::string_view command, const std::vector<std::string_view>& args)
{
	try
	{
		bool is_success = commands.at(command)(args);
		return is_success ? 0 : -1;
	}
	catch (...)
	{
		std::cerr << "Unexpected command.\n";
		return -1;
	}
}

int file_operations::execute(const std::string& args)
{
	std::istringstream iss{args};
	std::vector<std::string> split_args{std::istream_iterator<std::string>{iss}, {}};
	if (split_args.empty())  return false;
	return execute(split_args[0], { split_args.cbegin() + 1, split_args.cend() });
}

const std::map<std::string_view, std::function<file_operations::command_t>> file_operations::commands{
	{"copy", file_operations::copy}, { "move", file_operations::move },
	{ "info", file_operations::info }, { "mode", file_operations::mode },
	{ "--help", file_operations::help }, { "help", file_operations::help }
};