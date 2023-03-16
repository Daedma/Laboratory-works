

class ClientApplication
{
public:
	ClientApplication();

	int run(int argc, const char* const* const argv);

private:
	double input();

	void send();

	double receive();

	void output();
};
