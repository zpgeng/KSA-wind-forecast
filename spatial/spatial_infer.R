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
residual = ncvar_get(nc, "wind_residual_all_locations")
nc_close(nc)

mc_locations = readRDS('mc_locations.RDS')

coords = matrix(c(lon, lat),ncol = 2)

fit = NS_fit(
coords = coords,
data = residual[26280,],
cov.model = "matern", 
fit.radius = 2.5, 
mc.locations = mc_locations)

save(fit, file = "NS_fit.Rdata")