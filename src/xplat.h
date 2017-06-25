#pragma once

#include <stdlib.h>

double what_time_is_it_now();

void xplat_sleep(double seconds);

void* xplat_malloc(size_t count, size_t size);
