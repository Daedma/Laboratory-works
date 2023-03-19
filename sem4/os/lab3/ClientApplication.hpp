#pragma once

class ClientApplication
{
public:
	ClientApplication();

	int run(int argc, const char* const* argv);

	struct CalcParameters
	{
		double x;
		double accuracy;
	};

private:
	CalcParameters input();

	void send();

	CalcParameters receive();

	void output();
};
