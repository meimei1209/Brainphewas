# ============================================================
# ROC plotting with specified AUC values and DeLong tests
# ============================================================

library(pROC)

# ======== Path settings ========
# Prediction files should contain:
#   target_y
#   y_pred_idp
#   y_pred_cov
#   y_pred_idp_cov
input_dir <- "./Prediction/Phewas_prediction"

output_dir_roc  <- file.path(input_dir, "ROC_results")
output_dir_pval <- file.path(input_dir, "DeLong_results")

dir.create(output_dir_roc, showWarnings = FALSE, recursive = TRUE)
dir.create(output_dir_pval, showWarnings = FALSE, recursive = TRUE)

# ======== Diseases to plot ========
disease_keep <- c("290.1", "433", "272.1")

# ======== AUC labels from the evaluation script ========
# The AUC and 95% CI labels in the plotting script are pre-calculated using 
# 1000-iteration bootstrap resampling from Prediction_Bootstrap_Eval.py 
# to ensure robust interval estimation.
auc_label_map <- list(
  "290.1" = c(
    IDP = "0.700 [0.638 - 0.754]",
    Covariates = "0.656 [0.601 - 0.711]",
    IDP_Covariates = "0.717 [0.656 - 0.772]"
  ),
  "433" = c(
    IDP = "0.663 [0.645 - 0.681]",
    Covariates = "0.717 [0.700 - 0.734]",
    IDP_Covariates = "0.740 [0.725 - 0.756]"
  ),
  "272.1" = c(
    IDP = "0.651 [0.643 - 0.660]",
    Covariates = "0.718 [0.710 - 0.724]",
    IDP_Covariates = "0.721 [0.714 - 0.728]"
  )
)

# ======== Get prediction files for selected diseases ========
files <- file.path(input_dir, paste0(disease_keep, ".csv"))
files <- files[file.exists(files)]

cat("Detected", length(files), "selected files for ROC plotting and DeLong tests.\n")

# ======== Main loop ========
all_results <- data.frame()

for (f in files) {
  fname <- tools::file_path_sans_ext(basename(f))
  cat("\n>>> Processing file:", fname, "...\n")
  
  # ---------- Read prediction data ----------
  df <- tryCatch({
    read.csv(f)
  }, error = function(e) {
    cat("Failed to read file:", e$message, "\n")
    return(NULL)
  })
  if (is.null(df)) next
  
  # ---------- Check required columns ----------
  needed_cols <- c("target_y", "y_pred_idp", "y_pred_cov", "y_pred_idp_cov")
  if (!all(needed_cols %in% names(df))) {
    cat("Skipped: missing required prediction columns.\n")
    next
  }
  
  # ---------- Compute ROC curves for plotting and DeLong tests ----------
  roc_idp  <- roc(df$target_y, df$y_pred_idp, quiet = TRUE)
  roc_cov  <- roc(df$target_y, df$y_pred_cov, quiet = TRUE)
  roc_both <- roc(df$target_y, df$y_pred_idp_cov, quiet = TRUE)
  
  # ---------- DeLong tests ----------
  t1 <- roc.test(roc_idp, roc_cov,  method = "delong")
  t2 <- roc.test(roc_idp, roc_both, method = "delong")
  t3 <- roc.test(roc_cov, roc_both, method = "delong")
  
  # ---------- Store DeLong results ----------
  result <- data.frame(
    File = fname,
    Comparison = c(
      "IDP_vs_Covariates",
      "IDP_vs_IDP+Covariates",
      "Covariates_vs_IDP+Covariates"
    ),
    P_value = c(t1$p.value, t2$p.value, t3$p.value)
  )
  all_results <- rbind(all_results, result)
  
  # ---------- Use specified AUC labels from the evaluation script ----------
  auc_labels <- auc_label_map[[fname]]
  
  legend_labels <- c(
    paste0("IDP  AUC = ", auc_labels["IDP"]),
    paste0("Cov  AUC = ", auc_labels["Covariates"]),
    paste0("IDP + Cov  AUC = ", auc_labels["IDP_Covariates"])
  )
  
  p_labels <- c(
    paste0("P(IDP vs Cov) = ", format(t1$p.value, digits = 3, scientific = TRUE)),
    paste0("P(IDP vs IDP + Cov) = ", format(t2$p.value, digits = 3, scientific = TRUE)),
    paste0("P(Cov vs IDP + Cov) = ", format(t3$p.value, digits = 3, scientific = TRUE))
  )
  
  full_labels <- c(legend_labels, p_labels)
  full_cols <- c("#1B9E77", "#D95F02", "#7570B3", rep("black", 3))
  
  # ---------- Plot ROC curves ----------
  pdf_path <- file.path(output_dir_roc, paste0(fname, "_ROC.pdf"))
  pdf(pdf_path, width = 7, height = 6)
  par(mar = c(5, 5, 4, 2))
  
  plot(
    1 - roc_idp$specificities,
    roc_idp$sensitivities,
    type = "l",
    lwd = 2,
    col = "#1B9E77",
    xlim = c(0, 1),
    ylim = c(0, 1),
    xlab = "1 - Specificity",
    ylab = "Sensitivity",
    main = paste0(fname, " ROC"),
    cex.axis = 1.4,
    cex.lab  = 1.6,
    cex.main = 1.4
  )
  
  lines(1 - roc_cov$specificities, roc_cov$sensitivities, col = "#D95F02", lwd = 2)
  lines(1 - roc_both$specificities, roc_both$sensitivities, col = "#7570B3", lwd = 2)
  abline(0, 1, lty = 2, col = "gray60")
  
  legend(
    "bottomright",
    legend = full_labels,
    col = full_cols,
    lwd = c(rep(2, 3), rep(0, 3)),
    bty = "n",
    cex = 1.05,
    pt.cex = 1.05,
    text.col = "black"
  )
  
  dev.off()
  cat("Completed:", fname, "\n")
}

# ======== Save DeLong results ========
out_path <- file.path(output_dir_pval, "Selected_DeLong_results.csv")
write.csv(all_results, out_path, row.names = FALSE)

cat("\nAll selected files processed.\nDeLong results saved to:\n", out_path, "\n")