## install.packages("SIHR")
library(SIHR)
getAnywhere("compute_direction")

compute_direction <- function (loading, X, weight, deriv, mu = NULL, verbose = FALSE){
  n <- nrow(X)
  p <- ncol(X)
  loading.norm <- sqrt(sum(loading^2))
  if (loading.norm <= 1e-05) {
    message("loading norm too small, set proj direction as 0's \n")
    direction <- rep(0, length(loading))
  }
  else {
    if (n >= 6 * p) {
      temp <- sqrt(weight * deriv) * X
      Sigma.hat <- t(temp) %*% temp/n
      direction <- solve(Sigma.hat) %*% loading/loading.norm
    }
    else {
      direction_alter <- FALSE
      tryCatch(expr = {
        if (is.null(mu)) {
          if (n >= 0.9 * p) {
            step.vec <- incr.vec <- rep(NA, 3)
            for (t in 1:3) {
              index.sel <- sample(1:n, size = round(0.9 * 
                                                      p), replace = FALSE)
              Direction.Est.temp <- SIHR:::Direction_searchtuning(X[index.sel, 
                                                             , drop = F], loading, weight = weight[index.sel], 
                                                           deriv = deriv[index.sel])
              step.vec[t] <- Direction.Est.temp$step
              incr.vec[t] <- Direction.Est.temp$incr
            }
            step <- getmode(step.vec)
            incr <- getmode(incr.vec)
            Direction.Est <- Direction_fixedtuning(X, 
                                                   loading, weight = weight, deriv = deriv, 
                                                   step = step, incr = incr)
            while (Direction.Est$status != "optimal") {
              step <- step + incr
              Direction.Est <- Direction_fixedtuning(X, 
                                                     loading, weight = weight, deriv = deriv, 
                                                     step = step, incr = incr)
            }
            if (verbose) {
              cat(paste0("The projection direction is identified at mu = ", 
                         round(Direction.Est$mu, 6), "at step =", 
                         step, "\n"))
            }
          }
          else {
            Direction.Est <- SIHR:::Direction_searchtuning(X, 
                                                    loading, weight = weight, deriv = deriv)
            if (verbose) {
              cat(paste0("The projection direction is identified at mu = ", 
                         round(Direction.Est$mu, 6), "at step =", 
                         Direction.Est$step, "\n"))
            }
          }
        }
        else {
          Direction.Est <- Direction_fixedtuning(X, loading, 
                                                 weight = weight, deriv = deriv, mu = mu)
          while (Direction.Est$status != "optimal") {
            mu <- mu * 1.5
            Direction.Est <- Direction_fixedtuning(X, 
                                                   loading, weight = weight, deriv = deriv, 
                                                   mu = mu)
            if (verbose) 
              cat(paste0("The projection direction is identified at mu = ", 
                         round(Direction.Est$mu, 6), "\n"))
          }
        }
        direction <- Direction.Est$proj
      }, warning = function(w) {
        message("Caught an warning using CVXR!")
        print(w)
        direction_alter <<- TRUE
      }, error = function(e) {
        message("Caught an error using CVXR! Alternative method is applied for proj direction.")
        print(e)
        direction_alter <<- TRUE
      })
      if (direction_alter) {
        temp <- sqrt(weight * deriv) * X
        Sigma.hat <- t(temp) %*% temp/n
        Sigma.hat.inv <- diag(1/diag(Sigma.hat))
        direction <- Sigma.hat.inv %*% loading/loading.norm
      }
    }
  }
  return(direction)
}
assignInNamespace("compute_direction", compute_direction, ns = "SIHR")

run_dist_analysis <- function(z, X, z_data, X_data, U_list, G, A) {
  results_summary <- list()
  results_pvalue <- list()
  
  for(subtype in U_list) {
    print(subtype)
    fit <- Dist(z, X, z_data[[subtype]], X_data[[subtype]],
                G = G, A = A, model = "linear", tau = 0.5)
    
    s <- summary(fit)
    z_value <- s$output.est[5]$`z value`
    p_one_sided <- 1 - pnorm(z_value)
    
    results_summary[[subtype]] <- s
    results_pvalue[[subtype]] <- p_one_sided
  }
  
  return(list(summary = results_summary, pvalue = results_pvalue))
}


data_root <- "../data"
X_name <- "BRCA1"
Xz <- read.csv(file.path(data_root, "Internal_data", "InternalD_covariate_scaled_p200.csv"),row.names = 1)
XzE <- read.csv(file.path(data_root, "External_data", "ExternalD_covariate_scaled_p200.csv"),row.names = 1)
U_list <- c('Basal', 'Her2', 'LumA', 'LumB', 'Normal')
X_data <- list()
z_data <- list()
for (subtype in U_list) {
  Xzu <- read.csv(file.path(data_root, "Unlabel_data",sprintf("UnlabelD_covariate_%s_scaled_p200.csv", subtype)),row.names = 1)
  X_data[[subtype]] <- Xzu[[X_name]]
  z_data[[subtype]] <- as.matrix(Xzu[, !(names(Xzu) %in% X_name)])
}

X <- Xz[[X_name]]
XE <- XzE[[X_name]]
z <- as.matrix(Xz[, !(names(Xz) %in% X_name)])
zE <- as.matrix(XzE[, !(names(XzE) %in% X_name)])



p <- ncol(z)
G <- 1:p
A <- diag(p)


set.seed(1)
results_in <- run_dist_analysis(z, X, z_data, X_data, U_list, G, A)
adjusted_p <- p.adjust(unlist(results_in$pvalue), method = "BH")
results_in$pvalue_adjusted <- as.list(adjusted_p)

results_in$pvalue
# $Basal
# [1] 0.005430782
# 
# $Her2
# [1] 0.02241323
# 
# $LumA
# [1] 0.008601334
# 
# $LumB
# [1] 0.0006578849
# 
# $Normal
# [1] 3.13032e-05

results_in$pvalue_adjusted

# $Basal
# [1] 0.009051303
# 
# $Her2
# [1] 0.02241323
# 
# $LumA
# [1] 0.01075167
# 
# $LumB
# [1] 0.001644712
# 
# $Normal
# [1] 0.000156516

results_in$summary

# $Basal
# tau est.plugin est.debias Std. Error z value Pr(>|z|)  
# 0.5      1.121     0.6617     0.2598   2.547  0.01086 *
#   
# $Her2
# tau est.plugin est.debias Std. Error z value Pr(>|z|)  
# 0.5     0.5253     0.3306     0.1648   2.006  0.04483 *
#   
# $LumA
# tau est.plugin est.debias Std. Error z value Pr(>|z|)  
# 0.5      0.976     0.3927     0.1648   2.382   0.0172 *
#   
# $LumB
# tau est.plugin est.debias Std. Error z value Pr(>|z|)   
# 0.5     0.9993     0.5589      0.174   3.213 0.001316 **
#   
# $Normal
# tau est.plugin est.debias Std. Error z value  Pr(>|z|)    
# 0.5      2.024      1.179     0.2945   4.003 6.261e-05 ***






set.seed(101)

results_E <- run_dist_analysis(zE, XE, z_data, X_data, U_list, G, A)
adjusted_p <- p.adjust(unlist(results_E$pvalue), method = "BH")
results_E$pvalue_adjusted <- as.list(adjusted_p)

results_E$pvalue
# $Basal
# [1] 0.0002746299
# 
# $Her2
# [1] 0.04456773
# 
# $LumA
# [1] 0.0007059228
# 
# $LumB
# [1] 0.003215571
# 
# $Normal
# [1] 0.01092613

results_E$pvalue_adjusted
# $Basal
# [1] 0.001373149
# 
# $Her2
# [1] 0.04456773
# 
# $LumA
# [1] 0.001764807
# 
# $LumB
# [1] 0.005359285
# 
# $Normal
# [1] 0.01365767

results_E$summary
# $Basal
# tau est.plugin est.debias Std. Error z value  Pr(>|z|)    
# 0.5     0.6901     0.5588     0.1617   3.456 0.0005493 ***
#   
# $Her2
# tau est.plugin est.debias Std. Error z value Pr(>|z|)  
# 0.5     0.3351      0.206     0.1212     1.7  0.08914 .
# 
# $LumA
# tau est.plugin est.debias Std. Error z value Pr(>|z|)   
# 0.5     0.4415     0.2241    0.07021   3.192 0.001412 **
#   
# $LumB
# tau est.plugin est.debias Std. Error z value Pr(>|z|)   
# 0.5     0.3912      0.225    0.08258   2.725 0.006431 **
#   
# $Normal
# tau est.plugin est.debias Std. Error z value Pr(>|z|)  
# 0.5       1.02     0.4452     0.1942   2.293  0.02185 *