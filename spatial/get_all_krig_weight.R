### Set working directory to root/spatial
rm(list = ls())
setwd('spatial')

library(convoSPAT)
library(ncdf4)
library(fields)
library(StatMatch)

source('modified_convoSPAT.R')

nc = nc_open("../data/wind_residual_all_locations.nc")
lat = ncvar_get(nc,"lat_all_locations")
lon = ncvar_get(nc,"lon_all_locations")
nc_close(nc)

load('NS_model.Rdata')

nc = nc_open('../data/wind_farm_data.nc')
index_turbine_location = ncvar_get(nc,"index_turbine_location")  # index (staring from zero) of the 75 wind farms locations in the all 53333 locations
nc_close(nc)

pred.coords = matrix(c(lon, lat),ncol=2)

weight = Krig.weight(NS_model, pred.coords)

saveRDS(weight,"all_krig_weight.RDS")