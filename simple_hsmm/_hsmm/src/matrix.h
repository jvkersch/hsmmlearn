#ifndef ARRAY_MATRIX_H
	#define ARRAY_MATRIX_H

	#include <stdlib.h>
	#include <assert.h>



	template<class T>inline T** matrix(long Rows, long Columns)
	{
		T **m = new T*[Rows];
		if (m == NULL)
			return NULL;

		m[0] = new T[Rows * Columns];
		if(m[0] == NULL)
			return NULL;
		for(long i = 1; i < Rows; i++) 
			m[i] = m[i - 1] + Columns;

		return m;
	}


	template<class T>inline void free_matrix(T** Matrix)
	{
		delete[](Matrix[0]);
		delete[](Matrix);
	}

#endif
