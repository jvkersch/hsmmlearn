#ifndef ARRAY_CUBE_H
	#define ARRAY_CUBE_H

	#include <stdlib.h>
	#include <assert.h>
	#include "matrix.h"

	template<class T>inline T*** cube(long Depth, long Rows, long Columns)
	{
		T ***m = matrix<T*>(Depth, Rows);
		if (m == NULL)
			return NULL;

		m[0][0] = new T[Rows * Columns * Depth];
		if (m[0][0] == NULL)
			return NULL;

		for(long j = 1; j < Rows; j++)
			m[0][j] = m[0][j - 1] + Columns;

		for(long i = 1; i < Depth; i++)
		{
			m[i][0] = m[i - 1][0] + Rows * Columns;
			for(long j = 1; j < Rows; j++)
			m[i][j] = m[i][j - 1] + Columns;
		}

		return m;
	}



	template<class T>inline void free_cube(T*** Cube)
	{
		delete[](Cube[0][0]);
		free_matrix(Cube);
	}

#endif
