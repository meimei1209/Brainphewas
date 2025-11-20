############################################################
# Packages
############################################################
library(MendelR)
library(FastTraitR) 

############################################################
# Input: GWAS summary statistics
# ----------------------------------------------------------
# IDP:     vector of significant IDP GWAS summary filenames
# Outcome: vector of significant disease GWAS summary filenames
# GWAS summary files should be stored in the working directory
############################################################

############################################################
# Pre-MR analysis
# (Initial MR without removing confounder SNPs)
############################################################
mr_common_batch(IDP, Outcome, p1=5e-06, p2=5e-05, r2=0.01, kb=1000, 
                pop="EUR", no_plot = T, rm_F="F,10",steiger = F,no_clump=T,
                method_list=c("mr_ivw", "mr_egger_regression", 
                              "mr_weighted_median", "mr_weighted_mode",
                              "mr_wald_ratio"))

############################################################
# Identify confounder SNPs
############################################################
res =look_trait(file_name="IDP_SNP.csv", pval=1e-5)

############################################################
# Main MR analysis
# (Remove confounder SNPs and re-run MR)
############################################################
mr_common_batch(IDP, Outcome, p1=5e-06, p2=5e-05, r2=0.01, kb=1000, 
                pop="EUR", no_plot = T, rm_snps=confounderSNP, 
                rm_F="F,10",steiger = F,no_clump=T,
                method_list=c("mr_ivw", "mr_egger_regression", 
                              "mr_weighted_median", "mr_weighted_mode",
                              "mr_wald_ratio"))
