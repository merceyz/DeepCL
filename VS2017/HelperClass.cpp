#pragma once
#include "HelperClass.h"
#include <Windows.h>
#include <locale>
#include <codecvt>

std::string Helper::GetStartupPath()
{
    WCHAR buffer[MAX_PATH];
    GetModuleFileNameW(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::wstring(buffer).find_last_of(L"\\/");

    return WideStringToString(std::wstring(buffer).substr(0, pos));
}

std::string Helper::WideStringToString(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}
