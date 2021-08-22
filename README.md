# This repository contains the developed Python3 scripts to perform S-ESN

### Step 1. Download Data

1. The wind speed residual data at 3,173 knots

The wind speed residual data $Y_t(\mathbf{s}^\ast)$ at 3,173 knots from 2013 to 2016 (Feb. 29 in the leap year 2016 is removed) is stored as "wind_residual.nc" (netCDF4 file, 849 MB). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

(Optional) If you would like to see the S-ESN forecasts at all the 53,333 locations over Saudi Arabia, the two following datasets are needed.

2. The wind speed residual data at all 53,333 locations
The wind speed residual data $Y_t(\mathbf{s})$ at 53,333 knots from 2013 to 2016 (Feb. 29 in the leap year 2016 is removed) is stored as "wind_residual_all_locations.nc" (netCDF4 file, 14 GB). Please download it via https://repository.kaust.edu.sa/handle/10754/667127 and save it to the directory "./data".

  - **Due to data size limitation in the repository, the files to be downloaded are actually:**
    - wind_residual_all_locations.nc.partaa (5 GB)
    - wind_residual_all_locations.nc.partab (5 GB)
    - wind_residual_all_locations.nc.partac (4 GB)
  - Download all of them, save them to "./data" and run the following command to merge
  ```bash
  cat  wind_residual_all_locations.nc.part* > wind_residual_all_locations.nc
  ```

### Step 2. Run the Jupyter Notebook Wind_Forecast.ipynb
The Jupyter Notebook *Wind_Forecast.ipynb* contains the script to perform the S-ESN, where necessary comments are provided.

### List of files (directories)and description

| File or Directory | Description |
| :-------------:   |:-------------:|
| data | Directory containing needed data|
| src | Directory containing python source code for ESN |
| data/wind_residual.nc | Data of wind residuals at knots, to be downloaded |
| data/wind_residual_all_locations.nc | Data of wind residuals at all locations, to be downloaded |
| data/wind_farm_data.nc | Data of wind farm related information, to be downloaded |
| results/arima_wind_farm_predictions.npz | ARIMA results for 75 wind farms |
| results/quantiles_all.npz | Probabilistic forecasts for wind speed residuals at all 53,333 locations |
| results/quantiles_knots.npz | Probabilistic forecasts for wind speed residuals at 3,173 knots |
| results/quantiles_knots.npz | Probabilistic forecasts for wind speed residuals at 3,173 knots |
| spatial/esn_mse_all_locations.npz  | Precomputed MSE by S-ESN at all 53,333 locations |
| spatial/get_all_krig_weight.R  | Get the kriging weight from the nonstationary model for all 53,333 locations |
| spatial/get_wind_farm_krig_weight.R  | Get the kriging weight from the nonstationary model for 75 wind farm locations |
| spatial/mc_locations.RDS  | Locations for the 42 mixture components |
| spatial/modified_convoSPAT.R  | Modified R functions from the R package convoSPAT for spatial model inference |
| spatial/NS_fit.Rdata  | Precomputed inferential results for the nonstationary model |
| spatial/spatial_compile.R  | Compile the nonstationary model inferential results |
| spatial/spatial_infer.R  | Perform the nonstationary model inference |
| spatial/wind_farm_krig_weight.RDS  | Precomputed kriging weight from the nonstationary model for 75 wind farm locations |
| src/estimate_distribution.py  | Calibration code for probabilistic forecasts via quantiles |
| src/hyperpara.py  | Class for Hyperpara  |
| src/index.py | Wrapper for Index |
| src/model.py | Class of ESN model  | 
| src/utils.py | Supportive functions for drawing figures | 
| KSA_wind.ipynb | Jupyter Notebook to perform the S-ESN |