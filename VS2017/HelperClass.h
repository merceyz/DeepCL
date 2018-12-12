#pragma once
#include <string>

#define LoadKernel(fileName, kernelName) std::ifstream t; \
t.open(Helper::GetStartupPath() + "\\cl\\" + fileName, ios::binary | ios::in, _SH_DENYNO); \
 \
std::string data; \
\
std::string line = "";\
while (std::getline(t, line))\
{\
    data += line + "\n";\
}\
\
kernel = cl->buildKernelFromString(data, kernelName, options, "cl/forward1.cl");\

class Helper
{
public:
    static std::string GetStartupPath();
    static std::string WideStringToString(const std::wstring& wstr);

private:
    Helper();
};