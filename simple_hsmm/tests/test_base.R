library(hsmm);

pipar <- rep(1/3, 3)
tpmpar <- matrix(c(0, 0.5, 0.5,
                   0.7, 0, 0.3,
                   0.8, 0.2, 0), 3, byrow = TRUE)
rdpar <- list(np = matrix(c(0.1, 0.2, 0.3,
                            0.7, 0.6, 0.5,
                            0.2, 0.2, 0.2), 3, byrow = TRUE))
odpar <- list(mean = c(-1.5, 0, 1.5), var = c(0.5, 0.6, 0.8))

sim <- hsmm.sim(n = 2000, od = "norm", rd = "nonp",
                pi.par = pipar, tpm.par = tpmpar,
                rd.par = rdpar, od.par = odpar, seed = 3539)

write.csv(sim$obs, "viterbi_observations.csv", row.names = FALSE)

fit.vi <- hsmm.viterbi(sim$obs, od = "norm", rd = "nonp",
                       pi.par = pipar, tpm.par = tpmpar,
                       od.par = odpar, rd.par = rdpar)

write.csv(fit.vi$path, "viterbi_states.csv", row.names = FALSE)
