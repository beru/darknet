#include "xplat.h"

#ifdef __linux__

double what_time_is_it_now()
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

void xplat_sleep(double seconds)
{
    sleep(seconds);
}

#elif defined(_WIN32)

#pragma comment(lib, "Winmm.lib")
#include <Windows.h>

double what_time_is_it_now()
{
    DWORD now = timeGetTime();
    return now;
}

void xplat_sleep(double seconds)
{
    Sleep(seconds * 1000);
}

#endif

void* xplat_malloc(size_t count, size_t size)
{
static size_t total;
total += count * size;
printf("alloc %d total %d.\n", count * size / 1024, total / 1024);
    return calloc(count, size);
}
