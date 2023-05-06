int res = 0;
int d;
int size;
int* a;

int main()
{
	for (int i = 0; i != size; ++i)
	{
		if (a[i] < 0 && a[i] < d)
			res += a[i] * a[i] * a[i];
	}
}