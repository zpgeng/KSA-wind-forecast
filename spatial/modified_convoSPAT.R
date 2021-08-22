# All functions are drived from the R package, convoSPAT, which was 
# developed by Mark D. Risser (markdrisser@gmail.com)

library(convoSPAT)
library(fields)
library(StatMatch)

cov_spatial_ns_kappa <- function (Dist.mat, cov.model, pred.kappa = NULL, obs.kappa = NULL, symmetric = TRUE) 
{
    if (cov.model != "matern") {
        stop("Please specify a valid covariance model (matern)")
    }
    
    n = dim(Dist.mat)[1]
    m = dim(Dist.mat)[2]
    if(symmetric)
    {
        if(m != n ) stop("Dist.mat must be a symmetric matrix when symmetric is set TRUE")
        C = matrix(0,n,n)
        for(i in 1:(n-1))
            for(j in (i+1):n)
            {
                if(Dist.mat[i,j] < 1e-16)
                {
                    C[i,j] = 1
                    next
                }
                kappa = (obs.kappa[i] + obs.kappa[j]) / 2
                C[i,j] = (2**(1-kappa)) * Dist.mat[i,j]**kappa * besselK(Dist.mat[i,j], kappa) /  gamma(kappa)
            }
        C = C + t(C)
        diag(C) = 1
    }else
    {
        C = matrix(0,n,m)
        for(i in 1:n)
            for(j in 1:m)
            {
                if(Dist.mat[i,j] < 1e-16)
                {
                    C[i,j] = 1
                    next
                }
                kappa = (pred.kappa[i] + obs.kappa[j]) / 2
                C[i,j] = (2**(1-kappa)) * Dist.mat[i,j]**kappa * besselK(Dist.mat[i,j], kappa) /  gamma(kappa)
            }
    }
    return(C)
}

NS_fit <- function( coords = NULL, data = NULL,
                    cov.model = "matern", 
                    mc.locations = NULL,
                    fit.radius = NULL,
                    print.progress = TRUE,
                    local.pars.LB = NULL, local.pars.UB = NULL,
                    local.ini.pars = NULL){
  # Return the nonstationary local MLE
  if( is.null(fit.radius) ){message("\nPlease specify a fitting radius.\n")}
  #===========================================================================
  # Formatting for coordinates/data
  #===========================================================================
  if( is.null(coords) ) stop("Please provide a Nx2 matrix of spatial coordinates.")
  if( is.null(data) ) stop("Please provide a vector of observed data values.")

  coords <- as.matrix(coords)
  N <- dim(coords)[1]
  data <- as.matrix(data)
  p <- dim(data)[2]

  # Make sure cov.model is one of the permissible options
  if( cov.model != "matern")
    stop("Please specify a valid covariance model (matern).")

  K <- dim(mc.locations)[1]

  #===========================================================================
  # Check the mixture component locations
  #===========================================================================
  check.mc.locs <- mc_N( coords, mc.locations, fit.radius )
  message("\n-----------------------------------------------------------\n")
  message(paste("Fitting the nonstationary model: ", K, " local models with\nlocal sample sizes ranging between ", min(check.mc.locs), " and ", max(check.mc.locs), ".", sep = "" ))

  message("\nSpatially-varying nugget and spatially-varying variance.")
  message(paste("\nLocally anisotropic ", cov.model, " covariance.", sep = ""))
  
  if( min(check.mc.locs) < 5 ){message("\nWARNING: at least one of the mc locations has too few data points.\n")}
  message("\n-----------------------------------------------------------\n")

  #===========================================================================
  # Set the second nugget variance, if not specified
  #===========================================================================
  fixed.nugg2.var <- matrix(0,N,N)

  #===========================================================================
  # Specify lower, upper, and initial parameter values for optim()
  #===========================================================================
  lon_min <- min(coords[,1])
  lon_max <- max(coords[,1])
  lat_min <- min(coords[,2])
  lat_max <- max(coords[,2])

  max.distance <- sqrt( sum((c(lon_min,lat_min) - c(lon_max,lat_max))^2))

  # Lower limits for optim()
  lam1.LB <- 1e-05
  lam2.LB <- 1e-05
  tausq.local.LB <- 1e-05
  sigmasq.local.LB <- 1e-05
  kappa.local.LB <- 1e-05

  # Upper limits for optim()
  resid.var = var(data)
  lam1.UB <- max.distance/4
  lam2.UB <- max.distance/4
  tausq.local.UB <- 4*resid.var
  sigmasq.local.UB <- 4*resid.var
  kappa.local.UB <- 30
  
  # Initial values for optim()
  # Local estimation
  lam1.init <- max.distance/10
  lam2.init <- max.distance/10
  tausq.local.init <- 0.01*resid.var
  sigmasq.local.init <- 0.99*resid.var
  kappa.local.init <- 0.5
  
  #===========================================================================
  # Calculate the mixture component kernels (and other parameters) if not
  # user-specified
  #===========================================================================
  # Storage for the mixture component kernels
  mc.kernels <- array(NA, dim=c(2, 2, K))
  MLEs.save <- matrix(NA, K, 7)
  
  # Distance between coords and mc.locations
  coords_mc_dist <- rdist(coords, mc.locations)

  # Estimate the kernel function for each mixture component location,
  # completely specified by the kernel covariance matrix
  for( k in 1:K ){
    # Select local locations
    ind_local <- which(coords_mc_dist[,k] <= fit.radius)

    if(length(ind_local) > 900) ind_local = sample(ind_local, 900)
      
    # Subset
    temp.locations <- coords[ind_local,]
    n.fit <- dim(temp.locations)[1]
    temp.data <- as.matrix(data[ind_local,], nrow=n.fit)

    if( print.progress ){
      if(k == 1) message("Calculating the parameter set for:\n")
      message("mc location ", k,", using ", n.fit," observations...\n", sep="")
    }
    
    #####################################################
    # Local estimation, for covariance models with kappa
    # Locally anisotropic    
    f_loglik <- make_local_lik_without_mean( locations = temp.locations, 
                                          cov.model = cov.model,
                                          data = temp.data)
    MLEs <- optim( par = c(lam1.init, lam2.init, pi/4, tausq.local.init,
                           sigmasq.local.init, kappa.local.init ),
                   fn = f_loglik, method = "L-BFGS-B",
                   lower = c( lam1.LB, lam2.LB, 0, tausq.local.LB, sigmasq.local.LB,
                              kappa.local.LB),
                   upper = c( lam1.UB, lam2.UB, pi/2, tausq.local.UB, sigmasq.local.UB,
                              kappa.local.UB ) )
    covMLE <- MLEs$par


    # Warnings?
    if( MLEs$convergence != 0 ){
      if( MLEs$convergence == 52 ){
        message( paste("  There was a NON-FATAL error with optim(): \n  ",
                   MLEs$convergence, "  ", MLEs$message, "\n", sep = "") )
      }
      else{
        message( paste("  There was an error with optim(): \n  ",
                   MLEs$convergence, "  ", MLEs$message, "\n", sep = "") )
      }
    }

    # Save the kernel matrix
    mc.kernels[,,k] <- kernel_cov( covMLE[1:3] )

    # Save all MLEs
    # Parameter order: n, lam1, lam2, eta, tausq, sigmasq, kappa
    MLEs.save[k,] <- c( n.fit, covMLE )
  }

  # Put the MLEs into a data frame
  MLEs.save <- data.frame( MLEs.save )
  names(MLEs.save) <- c("n","lam1", "lam2", "eta", "tausq", "sigmasq", "kappa" )
    
  return(list(MLEs.save = MLEs.save, mc.kernels = mc.kernels))
}

get_NSconvo <- function(fit,
                    coords = NULL, data = NULL,
                    cov.model = "matern", 
                    mc.locations = NULL, lambda.w = NULL,
                    print.progress = TRUE,
                    local.pars.LB = NULL, local.pars.UB = NULL,
                    local.ini.pars = NULL){
  # Return the full NSconvo object for the data observations

  MLEs.save = fit$MLEs.save
  mc.kernels = fit$mc.kernels
  N <- dim(coords)[1]
  #===========================================================================
  # Calculate the weights for each observation location
  #===========================================================================
  # Weights
  weights.unnorm <- exp( -rdist(coords, mc.locations)^2/(2*lambda.w) )
  weights <- t(apply(X = weights.unnorm, MARGIN = 1, FUN = function(x){x/sum(x)}))

  #===========================================================================
  # Calculate the kernel ellipses and other spatially-varying quantities
  #===========================================================================
  obs.kernel11 <- rowSums( t(t(weights)*mc.kernels[1,1,]) )
  obs.kernel22 <- rowSums( t(t(weights)*mc.kernels[2,2,]) )
  obs.kernel12 <- rowSums( t(t(weights)*mc.kernels[1,2,]) )

  kernel.ellipses <- array(0, dim=c(2,2,N))
  kernel.ellipses[1,1,] <- obs.kernel11
  kernel.ellipses[1,2,] <- obs.kernel12
  kernel.ellipses[2,1,] <- obs.kernel12
  kernel.ellipses[2,2,] <- obs.kernel22

  #Calculate the spatially-varying nugget and variance
  obs.nuggets <- rowSums( t(t(weights)*MLEs.save$tausq) )
  obs.variance <- rowSums( t(t(weights)*MLEs.save$sigmasq) )
  obs.kappa <- rowSums( t(t(weights)*MLEs.save$kappa) )
  
  #===========================================================================
  # Calculate the nonstationary scale/distance matrices
  #===========================================================================
  arg11 <- obs.kernel11
  arg22 <- obs.kernel22
  arg12 <- obs.kernel12

  # Scale matrix
  det1 <- arg11*arg22 - arg12^2

  mat11_1 <- matrix(arg11, nrow = N) %x% matrix(1, ncol = N)
  mat11_2 <- matrix(1, nrow = N) %x% matrix(arg11, ncol = N)
  mat22_1 <- matrix(arg22, nrow = N) %x% matrix(1, ncol = N)
  mat22_2 <- matrix(1, nrow = N) %x% matrix(arg22, ncol = N)
  mat12_1 <- matrix(arg12, nrow = N) %x% matrix(1, ncol = N)
  mat12_2 <- matrix(1, nrow = N) %x% matrix(arg12, ncol = N)

  mat11 <- 0.5*(mat11_1 + mat11_2)
  mat22 <- 0.5*(mat22_1 + mat22_2)
  mat12 <- 0.5*(mat12_1 + mat12_2)

  det12 <- mat11*mat22 - mat12^2

  Scale.mat <- diag(sqrt(sqrt(det1))) %*% sqrt(1/det12) %*% diag(sqrt(sqrt(det1)))

  # Distance matrix
  inv11 <- mat22/det12
  inv22 <- mat11/det12
  inv12 <- -mat12/det12

  dists1 <- as.matrix(dist(coords[,1], upper = T, diag = T))
  dists2 <- as.matrix(dist(coords[,2], upper = T, diag = T))

  temp1_1 <- matrix(coords[,1], nrow = N) %x% matrix(1, ncol = N)
  temp1_2 <- matrix(1, nrow = N) %x% matrix(coords[,1], ncol = N)
  temp2_1 <- matrix(coords[,2], nrow = N) %x% matrix(1, ncol = N)
  temp2_2 <- matrix(1, nrow = N) %x% matrix(coords[,2], ncol = N)

  sgn.mat1 <- ( temp1_1 - temp1_2 >= 0 )
  sgn.mat1[sgn.mat1 == FALSE] <- -1
  sgn.mat2 <- ( temp2_1 - temp2_2 >= 0 )
  sgn.mat2[sgn.mat2 == FALSE] <- -1

  dists1.sq <- dists1^2
  dists2.sq <- dists2^2
  dists12 <- sgn.mat1*dists1*sgn.mat2*dists2

  Dist.mat <- sqrt( inv11*dists1.sq + 2*inv12*dists12 + inv22*dists2.sq )
    
  Data.Cov <- diag(sqrt(obs.variance)) %*% (Scale.mat * cov_spatial_ns_kappa( Dist.mat, 
                cov.model = cov.model, obs.kappa = obs.kappa )) %*% diag(sqrt(obs.variance))

  diag(Data.Cov) <- diag(Data.Cov) + obs.nuggets
  Data.Cov.chol <- chol(Data.Cov)

  #===========================================================================
  # Output
  #===========================================================================

  output <- list( mc.kernels = mc.kernels,
                  mc.locations = mc.locations,
                  MLEs.save = MLEs.save,
                  kernel.ellipses = kernel.ellipses,
                  data = data,
                  tausq.est = obs.nuggets,
                  sigmasq.est = obs.variance,
                  kappa.est = obs.kappa,
                  Cov.mat = Data.Cov,
                  Cov.mat.chol = Data.Cov.chol,
                  cov.model = cov.model,
                  coords = coords,
                  lambda.w = lambda.w
                  )

  class(output) <- "NSconvo"
  return(output)
}

make_local_lik_without_mean <- function( locations, cov.model, data){
  function(params) {
    # Locally anisotropic
    lam1 <- params[1]
    lam2 <- params[2]
    eta <- params[3]

    # For covariance models with kappa
    stopifnot(cov.model == "matern" || cov.model == "cauchy")
      
    # Estimate kappa and tausq
    tausq <- params[4]
    sigmasq <- params[5]
    kappa <- params[6]

    # Setup and covariance calculation
    N <- dim(locations)[1]
    m <- dim(data)[2]

    Pmat <- matrix(c(cos(eta), -sin(eta), sin(eta), cos(eta)), nrow = 2, byrow = T)
    Dmat <- diag(c(lam1, lam2))
    Sigma <- Pmat %*% Dmat %*% t(Pmat)
    distances <- mahalanobis.dist(data.x = locations, vc = Sigma)
    NS.cov <- sigmasq * cov_spatial(distances, cov.model = cov.model, cov.pars = c(1, 1), kappa = kappa)
    diag(NS.cov) = diag(NS.cov) + tausq

    possibleError <- tryCatch(
      cov.chol <- chol(NS.cov),
      error=function(e) e
    )

    # Check for error before calculating the likelihood
    if(inherits(possibleError, "error")){
      loglikelihood <- 1e+06
    } else{
      # Likelihood calculation
      tmp1 <- backsolve(cov.chol, data, transpose = TRUE)
      ResCinvRes <- t(tmp1) %*% tmp1
      loglikelihood <- m * sum(log(diag(cov.chol))) + 0.5 * sum(diag(ResCinvRes))
    }
    if (abs(loglikelihood) == Inf) { loglikelihood <- 1e+06 } # Make sure not Inf
    return(loglikelihood)
  }

}

Krig.weight <- function(object, pred.coords){
  if( !inherits(object, "NSconvo") ){
    stop("Object is not of type NSconvo.")
  }

  pred.coords <- as.matrix(pred.coords)

  M <- dim(pred.coords)[1] # 53333
  K <- dim(object$mc.locations)[1] # 42
  N <- length(object$sigmasq.est) # 3173

  #===========================================================================
  # Calculate prediction weights and parameters
  #===========================================================================
  # Weights
  pred.weights.unnorm <- exp( -rdist(pred.coords, object$mc.locations)^2/(2*object$lambda.w) )
  pred.weights <- t(apply(X = pred.weights.unnorm, MARGIN = 1, FUN = function(x){x/sum(x)}))

  # Calculate the kernel ellipses
  Pkern11 <- rowSums( t(t(pred.weights)*object$mc.kernels[1,1,]) )
  Pkern22 <- rowSums( t(t(pred.weights)*object$mc.kernels[2,2,]) )
  Pkern12 <- rowSums( t(t(pred.weights)*object$mc.kernels[1,2,]) )

  pred.nuggets <- rowSums( t(t(pred.weights) * object$MLEs.save$tausq) )

  pred.kappa = rowSums( t(t(pred.weights) * object$MLEs.save$kappa) )
  obs.kappa = object$kappa.est
    
  obs.variance <- object$sigmasq.est
  pred.variance <- rowSums( t(t(pred.weights) * object$MLEs.save$sigmasq) )

  message("-----------------------------------------------------------\n")
  message(paste("Calculating the ", M, " by ", N, " cross-correlation matrix.\n", sep=""))
  if( max(c(M,N)) > 1000 ){
    message("NOTE: this step can be take a few minutes.\n")
  }

  kern11 <- object$kernel.ellipses[1,1,]
  kern22 <- object$kernel.ellipses[2,2,]
  kern12 <- object$kernel.ellipses[1,2,]

  coords <- object$coords
  Pcoords <- pred.coords

  # Scale matrix
  det1 <- kern11*kern22 - kern12^2
  Pdet1 <- Pkern11*Pkern22 - Pkern12^2

  mat11_1 <- matrix(kern11, nrow = N) %x% matrix(1, ncol = M)
  mat11_2 <- matrix(1, nrow = N) %x% matrix(Pkern11, ncol = M)

  mat22_1 <- matrix(kern22, nrow = N) %x% matrix(1, ncol = M)
  mat22_2 <- matrix(1, nrow = N) %x% matrix(Pkern22, ncol = M)

  mat12_1 <- matrix(kern12, nrow = N) %x% matrix(1, ncol = M)
  mat12_2 <- matrix(1, nrow = N) %x% matrix(Pkern12, ncol = M)

  mat11 <- 0.5*(mat11_1 + mat11_2)
  mat22 <- 0.5*(mat22_1 + mat22_2)
  mat12 <- 0.5*(mat12_1 + mat12_2)

  det12 <- mat11*mat22 - mat12^2

  Scale.cross <- t(diag(sqrt(sqrt(det1))) %*% sqrt(1/det12) %*% diag(sqrt(sqrt(Pdet1))))

  # Distance matrix
  inv11 <- mat22/det12
  inv22 <- mat11/det12
  inv12 <- -mat12/det12

  dists1 <- rdist(coords[,1], Pcoords[,1])
  dists2 <- rdist(coords[,2], Pcoords[,2])

  temp1_1 <- matrix(coords[,1], nrow = N) %x% matrix(1, ncol = M)
  temp1_2 <- matrix(1, nrow = N) %x% matrix(Pcoords[,1], ncol = M)
  temp2_1 <- matrix(coords[,2], nrow = N) %x% matrix(1, ncol = M)
  temp2_2 <- matrix(1, nrow = N) %x% matrix(Pcoords[,2], ncol = M)

  sgn.mat1 <- ( temp1_1 - temp1_2 >= 0 )
  sgn.mat1[sgn.mat1 == FALSE] <- -1
  sgn.mat2 <- ( temp2_1 - temp2_2 >= 0 )
  sgn.mat2[sgn.mat2 == FALSE] <- -1

  dists1.sq <- dists1^2
  dists2.sq <- dists2^2
  dists12 <- sgn.mat1*dists1*sgn.mat2*dists2

  Dist.cross <- t(sqrt( inv11*dists1.sq + 2*inv12*dists12 + inv22*dists2.sq ))

  # Combine

  Unscl.cross <- cov_spatial_ns_kappa(Dist.cross, cov.model = "matern",
                                      pred.kappa = pred.kappa, obs.kappa = obs.kappa, symmetric = FALSE)
  NS.cross.corr <- Scale.cross * Unscl.cross

  CrossCov <- diag(sqrt(pred.variance)) %*% NS.cross.corr %*% diag(sqrt(obs.variance))
  crscov.Cinv <- t(backsolve(object$Cov.mat.chol, backsolve(object$Cov.mat.chol, t(CrossCov), transpose = TRUE)))
    
  return(crscov.Cinv)
}

