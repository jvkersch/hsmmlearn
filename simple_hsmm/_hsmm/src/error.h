#ifndef ERROR
	#define ERROR

	#include<string>
	#include<exception>

	using namespace std;

	class general_exception
	{
	public:
		general_exception() {}	
	};


	class var_nonpositive_exception : public general_exception
	{
	public:
		var_nonpositive_exception(): general_exception() {}
	};


	class memory_exception : public general_exception
	{
	public:
		memory_exception(): general_exception() {}
	};


	class file_exception : public general_exception
	{
	public:
		file_exception(): general_exception() {}
	};

#endif
