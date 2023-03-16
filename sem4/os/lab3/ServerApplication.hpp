
class ServerApplication
{
public:
	int run(int argc, const char* const* argv);

private:
	double receive();

	double calc();

	void send();

};