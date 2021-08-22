### Set working directory to root/spatial
rm(list = ls())
setwd('spatial')

library(convoSPAT)
library(ncdf4)
library(fields)
library(StatMatch)

source('modified_convoSPAT.R')

nc = nc_open("../data/wind_residual.nc")
lat_knots = ncvar_get(nc,"lat")
lon_knots = ncvar_get(nc,"lon")
nc_close(nc)

load('NS_fit.Rdata')

coords = matrix(c(lon_knots, lat_knots),ncol = 2)
mc_locations = readRDS('mc_locations.RDS')

NS_model = get_NSconvo(
fit,
coords = coords,
cov.model = "matern",  
lambda.w = 2,
mc.locations = mc_locations)

save(NS_model, file = "NS_model.Rdata")