__kernel void sortKernel(__global int *arrDevice, int j, int k, int size)
{
	unsigned int i = get_global_id(0);
	if (i >= size) return;
	unsigned int pt = i^j;

	if ((pt) > i) 
	{
		if ((i&k) == 0) 
		{
			if (arrDevice[i]>arrDevice[pt]) 
			{
				int val = arrDevice[i];
				arrDevice[i] = arrDevice[pt];
				arrDevice[pt] = val;
			}
		}
		if ((i&k) != 0) 
		{
			if (arrDevice[i]<arrDevice[pt]) 
			{
				int val = arrDevice[i];
				arrDevice[i] = arrDevice[pt];
				arrDevice[pt] = val;
			}
		}
	}
}