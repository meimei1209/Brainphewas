############################################################
# Mendelian Randomization analysis
############################################################

############################################################
# Packages
############################################################

library(MendelR)
library(FastTraitR)
library(MRlapPro)

############################################################
# Input: GWAS summary statistics
############################################################
# IDP:
#   A vector of GWAS summary statistic filenames for significant brain IDPs.
#
# Outcome:
#   A vector of GWAS summary statistic filenames for significant disease outcomes.
#
# UKB_Saige_Outcome:
#   A vector of GWAS summary statistic filenames for significant disease outcomes
#   from UK Biobank SAIGE analyses.
#
# Forward_Confounder_SNP:
#   A vector of confounder SNPs to be removed in forward MR.
#   These SNPs are provided in Supplementary Table S15.
#
# Reverse_Confounder_SNP:
#   A vector of confounder SNPs to be removed in reverse MR.
#   These SNPs are provided in Supplementary Table S16.
#
# All GWAS summary statistic files should be stored in the working directory.
############################################################

mr_methods <- c(
  "mr_ivw",
  "mr_egger_regression",
  "mr_weighted_median",
  "mr_weighted_mode",
  "mr_wald_ratio"
)

############################################################
# 1. Forward MR
############################################################

############################################################
# 1.1 Pre-MR analysis
# Initial MR without removing confounder SNPs
############################################################

mr_common_batch(
  id_exposures = IDP,
  id_outcomes  = Outcome,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR",
  no_plot = TRUE,
  rm_F = "F,10",
  steiger = FALSE,
  local_clump = TRUE,
  method_list = mr_methods
)

############################################################
# 1.2 Identify potential confounder SNPs
############################################################

potential_confounder_snps_forward <- look_trait(
  file_name = "IDP_SNP_5e6.csv",
  pval = 1e-5
)

############################################################
# 1.3 Main forward MR analysis
# Re-run MR after removing confounder SNPs
############################################################

mr_common_batch(
  id_exposures = IDP,
  id_outcomes  = Outcome,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR",
  no_plot = TRUE,
  rm_snps = Forward_Confounder_SNP,
  rm_F = "F,10",
  steiger = FALSE,
  local_clump = TRUE,
  method_list = mr_methods
)

############################################################
# 2. Reverse MR
############################################################

############################################################
# 2.1 Pre-MR analysis
# Initial reverse MR without removing confounder SNPs
############################################################

mr_common_batch(
  id_exposures = Outcome,
  id_outcomes  = IDP,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR",
  no_plot = TRUE,
  rm_F = "F,10",
  steiger = FALSE,
  local_clump = TRUE,
  method_list = mr_methods
)

############################################################
# 2.2 Main reverse MR analysis
# Re-run reverse MR after removing confounder SNPs
############################################################

mr_common_batch(
  id_exposures = Outcome,
  id_outcomes  = IDP,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR",
  no_plot = TRUE,
  rm_snps = Reverse_Confounder_SNP,
  rm_F = "F,10",
  steiger = FALSE,
  local_clump = TRUE,
  method_list = mr_methods
)

############################################################
# 3. MRlap sensitivity analysis
############################################################
# MRlap was used to account for sample overlap between UK Biobank
# imaging GWAS and UK Biobank SAIGE disease GWAS.
#
# Before running MRlap, confounder SNPs should be removed from the
# corresponding GWAS summary statistic files.
############################################################

############################################################
# 3.1 Forward MRlap
############################################################

run_MRlap(
  id_exposures = IDP,
  id_outcomes  = UKB_Saige_Outcome,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR"
)

############################################################
# 3.2 Reverse MRlap
############################################################

run_MRlap(
  id_exposures = UKB_Saige_Outcome,
  id_outcomes  = IDP,
  p1 = 5e-6,
  p2 = 5e-5,
  r2 = 0.01,
  kb = 1000,
  pop = "EUR"
)