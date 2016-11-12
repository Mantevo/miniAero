#include <stdio.h>
#include <sstream>
#include <fstream>

#include "MemoryUsage.h"

void get_memory_usage(size_t & current, size_t & high_water_mark)
{
    current = 0;
    high_water_mark = 0;
#ifdef BGQ_ENABLE_MEMSIZE
    uint64_t query_heap_used=-1;
    uint64_t query_heapmax=-1;

    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP,  &query_heap_used);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPMAX,  &query_heapmax);

    current = query_heap_used;
    high_water_mark = query_heapmax;
#else
    std::string vmrss, vmhwm;
    std::string line(128,'\0');
    std::ifstream proc_status("/proc/self/status");
    if (!proc_status) return;

    while ( vmrss.empty())
    {
        if(!std::getline(proc_status, line)) return;
        /* Find Current Usage */
        else if (line.substr(0, 6) == "VmRSS:")
        {
          vmrss = line.substr(7);
          std::istringstream iss(vmrss);
          iss >> current;
          current *= 1024;
        }
        /* Find High Water Mark */
        else if (line.substr(0, 6) == "VmHWM:")
        {
          vmhwm = line.substr(7);
          std::istringstream iss(vmhwm);
          iss >> high_water_mark;
          high_water_mark *= 1024;
        }
    }
#endif
}
