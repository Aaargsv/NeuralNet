#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <string>

class Timer {
public:
	Timer(const std::string &name):name(name), is_stop(false)
	{
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		if (!is_stop) {
			end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration = end - start;
			std::cout << name <<": ";
			std::cout << duration.count() << "s\n";
		}
	}

    void run()
	{
        start = std::chrono::high_resolution_clock::now();
    }

    void stop()
	{
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << name <<": ";
        std::cout << duration.count() << "s\n";
		is_stop = true;
    }

private:
	std::chrono::time_point<std::chrono::high_resolution_clock > start, end;
	std::string name;
	bool is_stop;
};

#endif //CNN_TIMER_H
