int a = 0, b = 0;
volatile int result = 0;

int main()
{
	if (a > b)
	{
		result = (a - b) / (a + b);
	}
	else if (a == b)
	{
		result = -a * b;
	}
	else if (a < b)
	{
		result = (3 * a - 2) / b;
	}
}