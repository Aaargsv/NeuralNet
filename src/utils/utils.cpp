#include "utils/utils.h"
#include <iostream>
#include <iomanip>


void print_table_element(const std::string &str, const int& width) {
    std::cout << std::left << std::setw(width) << std::setfill(' ') << str;
}