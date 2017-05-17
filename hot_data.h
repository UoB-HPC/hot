
#define ARCH_ROOT_PARAMS "../arch.params"
#define HOT_PARAMS "hot.params"

#define EPS 1.0e-10

typedef struct {
  double heat_capacity;
  double conductivity;
} HotData;

void initialise_hot_data(HotData* hot_data, const char* hot_params);

