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
#   - sex (Female / Male)
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
#           first diagnosis, death, 
#           or the end of hospital record availability (Oct 31, 2022)

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
  "age", "sex", "ethnicity", "BMI", "education", "TDI",
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

# Sex-specific diseases (exclude sex)
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

# Functional IDPs: variables starting with "node" or "edge"
func_cols <- grep("^(node|edge)", names(dat_image), value = TRUE, ignore.case = TRUE)
dat_image_func <- dat_image[, c("eid", func_cols), with = FALSE]

# Structural IDPs: all remaining columns
stru_cols <- setdiff(names(dat_image), c("eid", func_cols))
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
# PheWAS Stratified Analysis (by sex)
############################################################

## Structural IDPs
cox_res5 <- analyse_subgroup(
  dat_image_stru, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_stru,
  subgroup_factor = c("sex")
)
write_file(cox_res5, "phewas_stru_subgroup_sex.csv")

## Functional IDPs
cox_res6 <- analyse_subgroup(
  dat_image_func, phecode_nosex_data, cov_data,
  entry_time_threshold = 0,
  model = model_sex_func,
  subgroup_factor = c("sex")
)
write_file(cox_res6, "phewas_func_subgroup_sex.csv")

############################################################
# PheWAS Interaction Analysis (IDP × sex interaction)
############################################################

library(survival)
library(plyr)

# Build interaction variable
inter_data_sex <- cov_data[, c("eid", "sex"), with = FALSE]
inter_data_sex[, sex_interaction := dplyr::case_when(
  sex == "Female" ~ 1,
  sex == "Male"   ~ 0,
  TRUE ~ NA_real_
)]
inter_data_sex <- inter_data_sex[, c("eid", "sex_interaction"), with = FALSE]
cov_data_inter <- cov_data[, setdiff(names(cov_data), "sex"), with = FALSE]

# Directly reuse the previous covariate models that already exclude sex
model_sex_stru_inter <- model_sex_stru
model_sex_func_inter <- model_sex_func

# Interaction Cox function
analyse_cox_interaction_continuous <- function(
    exposure_data,
    outcome_data,
    cov_data,
    inter_data,
    entry_time_threshold = 0,
    model
) {
  library(dplyr)
  library(survival)
  library(plyr)
  
  results_all <- dplyr::data_frame()
  results_exposure <- dplyr::data_frame()
  results_model <- dplyr::data_frame()

  exposure_data <- as.data.frame(exposure_data)
  outcome_data  <- as.data.frame(outcome_data)
  cov_data      <- as.data.frame(cov_data)
  inter_data    <- as.data.frame(inter_data)
  
  if (exists(".convert_logic_to_integer", envir = asNamespace("FastUKB"))) {
    outcome_data <- get(".convert_logic_to_integer", envir = asNamespace("FastUKB"))(outcome_data)
  }
  
  colnames(exposure_data)[1] <- "f.eid"
  colnames(outcome_data)[1]  <- "f.eid"
  colnames(cov_data)[1]      <- "f.eid"
  colnames(inter_data)[1]    <- "f.eid"
  
  if (ncol(inter_data) != 2) {
    stop("Inter_data must contain exactly two columns: eid and one interaction variable.")
  }
  
  inter_name <- colnames(inter_data)[2]
  
  if (is.logical(inter_data[[2]])) {
    inter_data[[2]] <- as.integer(inter_data[[2]])
  }
  
  if (is.character(inter_data[[2]])) {
    stop("The interaction variable cannot be character. Please convert it to numeric, integer, or factor.")
  }
  
  total_exposures <- ncol(exposure_data) - 1
  
  for (j in 2:ncol(exposure_data)) {
    current_exposure_index <- j - 1
    expose_inf <- exposure_data[, c(1, j), drop = FALSE]
    expose_name <- names(expose_inf)[2]
    
    message("--------------------------------------------------")
    message(sprintf("Processing [%d/%d] Exposure: %s",
                    current_exposure_index, total_exposures, expose_name))
    
    if (is.factor(expose_inf[[2]])) {
      stop(sprintf("Exposure %s is factor. This function is only for continuous exposures.", expose_name))
    }
    if (!(is.numeric(expose_inf[[2]]) || is.integer(expose_inf[[2]]))) {
      stop(sprintf("Exposure %s is not numeric or integer. This function is only for continuous exposures.", expose_name))
    }
    
    is_log <- FALSE
    
    for (m_idx in seq_along(model)) {
      cov_vec <- unlist(model[[m_idx]])

      cov_vec_no_inter <- setdiff(cov_vec, inter_name)
      
      if (length(cov_vec_no_inter) == 0 || any(cov_vec_no_inter == "NA")) {
        formula_cov <- "NA"
        formula_str <- paste(
          "Surv(time, status==1) ~",
          paste0(expose_name, " * ", inter_name)
        )
      } else {
        formula_cov <- paste(cov_vec_no_inter, collapse = "+")
        formula_str <- paste(
          "Surv(time, status==1) ~",
          paste0(expose_name, " * ", inter_name),
          "+",
          formula_cov
        )
      }
      
      formula <- as.formula(formula_str)
      message("Model formula: ", formula_str)
      total_outcomes <- (ncol(outcome_data) - 1) %/% 2
      
      for (i in seq(2, (ncol(outcome_data) - 1), by = 2)) {
        current_outcome_index <- i / 2
        
        outcome_inf <- outcome_data[, c(1, i, i + 1), drop = FALSE]
        outcome_base_name <- sub("_time$", "", colnames(outcome_inf)[2])
        
        message(sprintf("Processing [%d/%d] Outcome: %s",
                        current_outcome_index, total_outcomes, outcome_base_name))
        
        colnames(outcome_inf)[2:3] <- c("status", "time")
        
        participant <- outcome_inf %>%
          merge(expose_inf, by = "f.eid", all = TRUE) %>%
          merge(cov_data, by = "f.eid", all = TRUE) %>%
          merge(inter_data, by = "f.eid", all = TRUE)
        
        n_p1 <- nrow(participant)
        participant <- participant %>% filter(time >= entry_time_threshold)
        n_p2 <- nrow(participant)
        participant <- participant %>% na.omit()
        n_p3 <- nrow(participant)
        
        difference_baseline <- n_p1 - n_p2
        difference_missing  <- n_p2 - n_p3
        
        if (!is_log) {
          message("")
          message(sprintf("Total number of participants: %d", n_p1))
          message(sprintf("After baseline exclusion: %d; Difference: %d", n_p2, difference_baseline))
          message(sprintf("After missing value exclusion: %d; Difference: %d", n_p3, difference_missing))
          message("")
          is_log <- TRUE
        }
        
        if (n_p3 == 0) {
          warning(sprintf("All data excluded for exposure %s and outcome %s.", expose_name, outcome_base_name))
          next
        }
        
        case_n <- sum(as.numeric(as.character(participant$status)))
        
        if (case_n < 20) {
          
          results_a <- data.frame(
            expose = expose_name,
            outcome = names(outcome_data)[i],
            Case_N = case_n,
            Control_N = length(participant$status) - case_n,
            Person_years = sum(participant$time),
            HR = NA_real_,
            `95%_CI` = NA_character_,
            `P value` = NA_real_,
            beta_interaction = NA_real_,
            HR_interaction = NA_real_,
            `95%_CI_interaction` = NA_character_,
            `P_interaction` = NA_real_,
            formula = formula_str,
            class = "continue",
            stringsAsFactors = FALSE
          )
          
          results_exposure <- plyr::rbind.fill(results_exposure, results_a)
          
          next
        }
        
        fit <- tryCatch(
          survival::coxph(formula = formula, data = participant),
          error = function(e) {
            message("Model failed:")
            message("  exposure = ", expose_name)
            message("  outcome  = ", outcome_base_name)
            message("  formula  = ", formula_str)
            message("  error    = ", e$message)
            return(NULL)
          }
        )
        
        if (is.null(fit)) {
          results_a <- data.frame(
            expose = expose_name,
            outcome = names(outcome_data)[i],
            Case_N = case_n,
            Control_N = length(participant$status) - case_n,
            Person_years = sum(participant$time),
            HR = NA_real_,
            `95%_CI` = NA_character_,
            `P value` = NA_real_,
            beta_interaction = NA_real_,
            HR_interaction = NA_real_,
            `95%_CI_interaction` = NA_character_,
            `P_interaction` = NA_real_,
            formula = formula_str,
            class = "continue",
            stringsAsFactors = FALSE
          )
          results_exposure <- plyr::rbind.fill(results_exposure, results_a)
          next
        }
        
        fit_sum <- summary(fit)
        coef_tab <- as.data.frame(fit_sum$coefficients)
        coef_tab$term <- rownames(coef_tab)
        
        main_row <- coef_tab[coef_tab$term == expose_name, , drop = FALSE]
        
        inter_row <- coef_tab[
          coef_tab$term %in% c(
            paste0(expose_name, ":", inter_name),
            paste0(inter_name, ":", expose_name)
          ),
          ,
          drop = FALSE
        ]
        
        if (nrow(main_row) == 0) {
          HR_main <- NA_real_
          CI_main <- NA_character_
          P_main  <- NA_real_
        } else {
          LCI_main <- exp(main_row$coef - 1.96 * main_row$`se(coef)`)
          UCI_main <- exp(main_row$coef + 1.96 * main_row$`se(coef)`)
          HR_main  <- main_row$`exp(coef)`
          CI_main  <- paste0(
            sprintf("%.4f", round(LCI_main, 4)),
            ", ",
            sprintf("%.4f", round(UCI_main, 4))
          )
          P_main <- main_row$`Pr(>|z|)`
        }
        
        if (nrow(inter_row) == 0) {
          beta_inter <- NA_real_
          HR_inter   <- NA_real_
          CI_inter   <- NA_character_
          P_inter    <- NA_real_
        } else {
          LCI_inter <- exp(inter_row$coef - 1.96 * inter_row$`se(coef)`)
          UCI_inter <- exp(inter_row$coef + 1.96 * inter_row$`se(coef)`)
          beta_inter <- inter_row$coef
          HR_inter   <- exp(inter_row$coef)
          CI_inter   <- paste0(
            sprintf("%.4f", round(LCI_inter, 4)),
            ", ",
            sprintf("%.4f", round(UCI_inter, 4))
          )
          P_inter <- inter_row$`Pr(>|z|)`
        }
        
        results_a <- data.frame(
          expose = expose_name,
          outcome = names(outcome_data)[i],
          Case_N = case_n,
          Control_N = length(participant$status) - case_n,
          Person_years = sum(participant$time),
          HR = HR_main,
          `95%_CI` = CI_main,
          `P value` = P_main,
          beta_interaction = beta_inter,
          HR_interaction = HR_inter,
          `95%_CI_interaction` = CI_inter,
          `P_interaction` = P_inter,
          formula = formula_str,
          class = "continue",
          stringsAsFactors = FALSE
        )
        
        results_exposure <- plyr::rbind.fill(results_exposure, results_a)
      }
      
      if (m_idx == 1) {
        results_model <- results_exposure
        new_column_names <- c(
          "model_1 HR",
          "model_1 95%_CI",
          "model_1 P value",
          "model_1 beta_interaction",
          "model_1 HR_interaction",
          "model_1 95%_CI_interaction",
          "model_1 P_interaction",
          "model_1 formula"
        )
        names(results_model)[6:13] <- new_column_names
      } else {
        tmp <- results_exposure[, 6:13, drop = FALSE]
        model_name <- paste0("model_", m_idx)
        names(tmp) <- paste(
          model_name,
          c("HR", "95%_CI", "P value",
            "beta_interaction", "HR_interaction", "95%_CI_interaction", "P_interaction", "formula")
        )
        results_model <- cbind_fill(results_model, tmp)
      }
      
      results_exposure <- dplyr::data_frame()
    }
    
    results_all <- plyr::rbind.fill(results_all, results_model)
    results_model <- dplyr::data_frame()
  }
  
  return(results_all)
}

# Structural IDPs interaction analysis
cox_inter_res_stru <- analyse_cox_interaction_continuous(
  exposure_data = dat_image_stru,
  outcome_data = phecode_nosex_data,
  cov_data = cov_data_inter,
  inter_data = inter_data_sex,
  entry_time_threshold = 0,
  model = model_sex_stru_inter
)

write_file(cox_inter_res_stru, "cox_inter_stru.csv")

# Functional IDPs interaction analysis
cox_inter_res_func <- analyse_cox_interaction_continuous(
  exposure_data = dat_image_func,
  outcome_data = phecode_nosex_data,
  cov_data = cov_data_inter,
  inter_data = inter_data_sex,
  entry_time_threshold = 0,
  model = model_sex_func_inter
)

write_file(cox_inter_res_func, "cox_inter_func.csv")