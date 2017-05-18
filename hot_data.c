#include "hot_data.h"
#include "../params.h"

void initialise_hot_data(HotData* hot_data, const char* hot_params)
{
  hot_data->conductivity = get_double_parameter("conductivity", hot_params);
  hot_data->heat_capacity = get_double_parameter("heat_capacity", hot_params);
}

