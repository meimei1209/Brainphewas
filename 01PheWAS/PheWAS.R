############################################################
# Packages
############################################################

library(FastUKB)
library(dplyr)
library(data.table)

############################################################
# Data Preparation
############################################################
# IDP data: dat_image
# Contains:
#   - eid
#   - all standardized imaging-derived phenotypes (IDPs)

# Covariate data: cov_data
# Contains:
#   - eid
#   - age
#   - gender (female / male)
#   - ethnicity (white / non-white)
#   - education (college / other levels / unknown)
#   - UK_assessment_centre (Bristol / Cheadle / Newcastle / Reading)
#   - BMI (underweight / normal / overweight / obese)
#   - TDI (Townsend deprivation index)
#   - smoking_status (never / previous / current)
#   - drinking_status (never / previous / current)
#   - total_intracranial_volume
#   - rfMRI_head_motion
#   - rfMRI_signal_to_noise

# Disease data: phecode_data
# Contains:
#   - eid
#   - status: all disease status for each phecode (1 = incident disease, 0 = no disease)
#   - time: all disease follow-up time from imaging date until:
#           first diagnosis, death, loss to follow-up, 
#           or end of hospital record updates (Oct 31, 2022)

############################################################
# Define Sex-Related Phecodes
############################################################

sex_related_prefixes <- as.character(c(
  174, 175, 180, 182, 184, 185, 187, 188, 218, 220, 221, 222, 256, 257,
  600, 601, 602, 603, 604, 605, 608, 609,610, 611, 612, 613, 614, 615, 
  617, 618, 619, 620, 621,622, 623, 624, 625, 626, 627, 628, 634, 635, 
  636, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 652, 653, 
  654, 655, 658, 669, 671, 674, 751.11, 751.12, 792, 796
))

# Extract column names for sex-specific and non–sex-specific diseases
all_cols <- colnames(phecode_data)
sex_cols <- all_cols[sapply(all_cols, \(x) any(startsWith(x, sex_related_prefixes)))]

phecode_sex_data   <- phecode_data[, c("eid", sex_cols), with = FALSE]
phecode_nosex_data <- phecode_data[, setdiff(all_cols, sex_cols), with = FALSE]

############################################################
# Covariate Models
############################################################

# Non–sex-specific diseases
model_stru <- c(
  "age", "gender", "ethnicity", "BMI", "education", "TDI",
  "smoking_status", "drinking_status", "UK_assessment_centre",
  "total_intracranial_volume"
)  # For T1 / DWI models

model_func <- c(
  model_stru,
  "rfMRI_head_motion",
  "rfMRI_signal_to_noise"
)  # For fMRI models

model_nosex_stru <- list(model_stru)
model_nosex_func <- list(model_func)

# Sex-specific diseases (exclude gender)
model_stru_sex <- c(
  "age", "ethnicity", "BMI", "education", "TDI",
  "smoking_status", "drinking_status", "UK_assessment_centre",
  "total_intracranial_volume"
)

model_func_sex <- c(
  model_stru_sex,
  "rfMRI_head_motion",
  "rfMRI_signal_to_noise"
)

model_sex_stru <- list(model_stru_sex)
model_sex_func <- list(model_func_sex)

############################################################
# Build Structural and Functional IDP Matrices
############################################################

# Functional IDPs: variables starting with "nodes" or "edges"
func_cols <- grep("^(nodes|edges)", names(dat_image), value = TRUE)
dat_image_func <- dat_image[, c("eid", func_cols), with = FALSE]

# Structural IDPs: all remaining columns
stru_cols <- setdiff(names(dat_image), func_cols)
dat_image_stru <- dat_image[, c("eid", stru_cols), with = FALSE]

############################################################
# PheWAS Analysis
############################################################

## Structural IDPs
### Non–sex-specific diseases
cox_res1 <- analyse_cox(
  dat_image_stru, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_nosex_stru
)
write_file(cox_res1, "phewas_stru_non_sex.csv")

### Sex-specific diseases
cox_res2 <- analyse_cox(
  dat_image_stru, phecode_sex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_stru
)
write_file(cox_res2, "phewas_stru_sex.csv")

## Functional IDPs
### Non–sex-specific diseases
cox_res3 <- analyse_cox(
  dat_image_func, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_nosex_func
)
write_file(cox_res3, "phewas_func_non_sex.csv")

### Sex-specific diseases
cox_res4 <- analyse_cox(
  dat_image_func, phecode_sex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_func
)
write_file(cox_res4, "phewas_func_sex.csv")

############################################################
# PheWAS Subgroup Analysis (by sex)
############################################################

## Structural IDPs
cox_res5 <- analyse_subgroup(
  dat_image_stru, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_stru,
  subgroup_factor = c("gender")
)
write_file(cox_res5, "phewas_stru_subgroup_sex.csv")

## Functional IDPs
cox_res6 <- analyse_subgroup(
  dat_image_func, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_func,
  subgroup_factor = c("gender")
)
write_file(cox_res6, "phewas_func_subgroup_sex.csv")
