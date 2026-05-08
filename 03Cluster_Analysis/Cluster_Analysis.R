############################################################
# Packages
############################################################
library(data.table)
library(dplyr)
library(tidyr)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(stringr)
library(forcats)
library(RColorBrewer)
library(grid)
library(cluster)
library(factoextra)
library(patchwork)

############################################################
# Load PheWAS summary results and data preparation
############################################################
# dat contains:
#   exposure     – IDP name
#   Case_N       – number of incident cases
#   Control_N    – number of controls
#   Person_years – total follow-up time
#   HR           – hazard ratio from Cox model
#   CI95         – 95% confidence interval
#   p            – P value
#   phenotype    – phecode
#   Outcome      – disease name
#   Outcomegroup – disease category

# Keep only outcomes with Case_N ≥ 20
dat <- dat %>%
  filter(Case_N >= 20)

############################################################
# Remove congenital and pregnancy-related diseases
############################################################

dat <- dat %>%
  filter(Outcomegroup != "Congenital") %>%
  filter(Outcomegroup != "Pregnancy")

############################################################
# Add IDP category
############################################################

dat <- dat %>%
  mutate(
    exposuregroup = case_when(
      str_starts(exposure, "Mean")   ~ "Cortical Thickness",
      str_starts(exposure, "Area")   ~ "Cortical Surface Area",
      str_starts(exposure, "Volume") ~ "Cortical Volume",
      str_starts(exposure, "MD")     ~ "White Matter MD",
      str_starts(exposure, "FA")     ~ "White Matter FA",
      str_starts(exposure, "Node")   ~ "Functional Network Nodes",
      str_starts(exposure, "Edge")   ~ "Functional Connectivity Edges",
      TRUE                           ~ "Subcortical Volume"
    ),
    exposure = if_else(
      exposuregroup == "Subcortical Volume",
      paste0("Volume of ", exposure),
      exposure
    )
  )

############################################################
# Define Bonferroni threshold
############################################################

n_exp  <- dplyr::n_distinct(dat$exposure)
n_out  <- dplyr::n_distinct(dat$Outcome)
n_test <- n_exp * n_out
alpha  <- 0.05
bonf_thr <- alpha / n_test
sig_thr <- -log10(bonf_thr)

dat_sig <- dat %>%
  filter(p < bonf_thr)

keep_exposures <- unique(dat_sig$exposure)
keep_outcomes  <- unique(dat_sig$Outcome)


############################################################
# Metadata and signed -log10(P) matrix
############################################################

brain_meta <- dat %>% distinct(exposure, exposuregroup)

dis_meta <- dat %>% distinct(Outcome, Outcomegroup)

dat2 <- dat %>%
  mutate(
    beta    = log(HR),
    sign    = if_else(beta >= 0, 1, -1),
    mlog10p = -log10(p),
    S       = sign * mlog10p
  )

M_all <- dat2 %>%
  filter(
    exposure %in% keep_exposures,
    Outcome %in% keep_outcomes
  ) %>%
  select(exposure, Outcome, S) %>%
  pivot_wider(names_from = Outcome, values_from = S) %>%
  as.data.frame()

rownames(M_all) <- M_all$exposure
M_all$exposure <- NULL

############################################################
# Color schemes
############################################################

ImageClass_colors <- c(
  "Cortical Thickness"            = "#377EB8",
  "Cortical Surface Area"         = "#E41A1C",
  "Cortical Volume"               = "#999999",
  "Subcortical Volume"            = "#A65628",
  "White Matter FA"               = "#F781BF",
  "White Matter MD"               = "#A6CEE3",
  "Functional Network Nodes"      = "#FF7F00",
  "Functional Connectivity Edges" = "#984EA3"
)

diseaseclass_colors <- c(
  "Circulatory"             = "#A6CEE3",
  "Dermatologic"            = "#7570B3",
  "Digestive"               = "#1F78B4",
  "Endocrine and metabolic" = "#66A61E",
  "Genitourinary"           = "#E6AB02",
  "Hematologic and immune"  = "#A6761D",
  "Infectious"              = "#666666",
  "Musculoskeletal"         = "#E7298A",
  "Neoplasms"               = "#B2DF8A",
  "Nervous"                 = "#FB9A99",
  "Ophthalmic and ENT"      = "#1B9E77",
  "Psychiatric"             = "#CAB2D6",
  "Respiratory"             = "#FF7F00",
  "Symptomatic"             = "#6A3D9A",
  "Traumatic and toxic"     = "#B15928"
)

col_fun <- colorRamp2(
  c(-10, -5, -1, 0, 1, 5, 10),
  c("#0B8FA8", "#5DAFC0", "#DCEAEC", "#F2F2F2",
    "#F8D7CF", "#F59A86", "#F04E4E")
)

############################################################
# Row-wise clustering, brain-driven clusters
############################################################

outcome_order <- c(
  "Infectious",
  "Neoplasms",
  "Endocrine and metabolic",
  "Hematologic and immune",
  "Psychiatric",
  "Nervous",
  "Ophthalmic and ENT",
  "Circulatory",
  "Respiratory",
  "Digestive",
  "Genitourinary",
  "Dermatologic",
  "Musculoskeletal",
  "Symptomatic",
  "Traumatic and toxic"
)

M_row <- M_all

dis_meta$Outcomegroup <- factor(dis_meta$Outcomegroup, levels = outcome_order)

outcome_order_row <- dis_meta %>%
  filter(Outcome %in% colnames(M_row)) %>%
  arrange(factor(Outcomegroup, levels = outcome_order)) %>%
  pull(Outcome)

M_row <- M_row[, outcome_order_row, drop = FALSE]

brain_meta2 <- brain_meta[match(rownames(M_row), brain_meta$exposure), , drop = FALSE]
dis_meta2   <- dis_meta[match(colnames(M_row), dis_meta$Outcome), , drop = FALSE]

d_row <- as.dist(1 - cor(t(M_row), method = "pearson"))
hc_row <- hclust(d_row, method = "ward.D2")

############################################################
# Select number of row clusters
############################################################

p1_row <- fviz_nbclust(
  M_row,
  hcut,
  method = "wss",
  hc_method = "ward.D2",
  hc_metric = "pearson"
) +
  labs(
    title = "Elbow Method (Exposure clustering)",
    x = "Number of clusters (k)",
    y = "Total within-cluster sum of squares"
  ) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 14, color = "black"),
    axis.title.y = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 12, color = "black"),
    axis.text.y = element_text(size = 12, color = "black")
  )

k_candidates <- 2:15
sil_width_row <- numeric(max(k_candidates))

for (k in k_candidates) {
  cluster_k <- cutree(hc_row, k = k)
  sil <- silhouette(cluster_k, d_row)
  sil_width_row[k] <- mean(sil[, 3])
}

df_row <- data.frame(
  k = k_candidates,
  silhouette = sil_width_row[k_candidates]
)

p2_row <- ggplot(df_row, aes(x = k, y = silhouette)) +
  geom_line(color = "#5181B1") +
  geom_point(color = "#5181B1") +
  scale_x_continuous(breaks = k_candidates) +
  labs(
    title = "Silhouette Method (Exposure clustering)",
    x = "Number of clusters (k)",
    y = "Average silhouette width"
  ) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 14, color = "black"),
    axis.title.y = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 12, color = "black"),
    axis.text.y = element_text(size = 12, color = "black")
  )

ggsave(
  filename = "Exposure_Elbow_Silhouette.pdf",
  plot = p2_row + p1_row,
  width = 10,
  height = 5
)

# Final k was selected based on elbow plots, silhouette analysis,
# preservation of association patterns, and biological interpretability.
optimal_k_row <- 8
k_row <- optimal_k_row

row_clusters <- cutree(hc_row, k = k_row)

row_cluster_df <- data.frame(
  exposure = names(row_clusters),
  row_cluster = factor(row_clusters)
) %>%
  left_join(brain_meta, by = "exposure")

write.csv(row_cluster_df, "row_clusters_exposure.csv", row.names = FALSE)

############################################################
# Row-cluster heatmap
############################################################

row_cluster_colors <- c(
  brewer.pal(12, "Paired"),
  brewer.pal(8, "Set2")
)[1:k_row]

row_anno2 <- rowAnnotation(
  ImageClass = brain_meta2$exposuregroup,
  RowCluster = factor(row_clusters),
  col = list(
    ImageClass = ImageClass_colors,
    RowCluster = structure(
      row_cluster_colors,
      names = levels(factor(row_clusters))
    )
  )
)

col_anno2 <- HeatmapAnnotation(
  DiseaseClass = dis_meta2$Outcomegroup,
  col = list(DiseaseClass = diseaseclass_colors)
)

lgd_star <- Legend(
  labels = "Significant",
  type = "graphics",
  graphics = list(function(x, y, w, h) {
    grid.text(
      "*",
      x,
      y,
      gp = gpar(fontsize = 12, col = "black", fontface = "bold")
    )
  })
)

pdf("Heatmap_Bonferroni_RowCluster_sig_star.pdf", 11, 12)
draw(
  Heatmap(
    M_row,
    name = "S",
    col = col_fun,
    cluster_rows = as.dendrogram(hc_row),
    cluster_columns = FALSE,
    show_row_names = FALSE,
    show_column_names = FALSE,
    right_annotation = row_anno2,
    top_annotation = col_anno2,
    cell_fun = function(j, i, x, y, w, h, fill) {
      if (abs(M_row[i, j]) >= sig_thr) {
        grid.text(
          "*",
          x = x,
          y = y - unit(0.0045, "snpc"),
          gp = gpar(fontsize = 10, col = "black"),
          just = "centre"
        )
      }
    }
  ),
  annotation_legend_list = list(lgd_star)
)
dev.off()

############################################################
# Sub-cluster heatmap function
############################################################

plot_row_cluster_heatmap <- function(
    cluster_id,
    width = 10,
    height = 7,
    padding = unit(c(3, 2, 1.5, 5), "cm"),
    wrap_colnames = FALSE,
    wrap_width = 70,
    star_offset = 0.01,
    representative = FALSE,
    min_sig_per_outcome = 8
) {
  exp_in_cl <- row_cluster_df %>%
    filter(row_cluster == cluster_id) %>%
    pull(exposure)
  
  dis_in_cl <- dat2 %>%
    filter(exposure %in% exp_in_cl, p < bonf_thr) %>%
    pull(Outcome) %>%
    unique()
  
  if (length(exp_in_cl) == 0 || length(dis_in_cl) == 0) {
    message("Cluster ", cluster_id, " has no exposure or significant outcome.")
    return(NULL)
  }
  
  subM_df <- dat2 %>%
    filter(exposure %in% exp_in_cl, Outcome %in% dis_in_cl) %>%
    select(exposure, Outcome, S) %>%
    pivot_wider(names_from = Outcome, values_from = S) %>%
    as.data.frame()
  
  rownames(subM_df) <- subM_df$exposure
  subM_df$exposure <- NULL
  
  subM_df[] <- lapply(subM_df, function(x) suppressWarnings(as.numeric(x)))
  M_for_heat <- as.matrix(subM_df)
  storage.mode(M_for_heat) <- "double"
  M_for_heat <- M_for_heat[exp_in_cl, dis_in_cl, drop = FALSE]
  
  d_row_sub <- as.dist(
    1 - cor(t(M_for_heat), method = "pearson", use = "pairwise.complete.obs")
  )
  d_col_sub <- as.dist(
    1 - cor(M_for_heat, method = "pearson", use = "pairwise.complete.obs")
  )
  
  hc_row_sub <- hclust(d_row_sub, method = "ward.D2")
  hc_col_sub <- hclust(d_col_sub, method = "ward.D2")
  
  if (representative) {
    keep_cols <- colSums(abs(M_for_heat) >= sig_thr, na.rm = TRUE) >= min_sig_per_outcome
    M_for_heat <- M_for_heat[, keep_cols, drop = FALSE]
    
    if (ncol(M_for_heat) == 0) {
      message("No representative outcomes in cluster ", cluster_id)
      return(NULL)
    }
    
    M_for_heat <- t(M_for_heat)
    cluster_rows <- FALSE
    cluster_columns <- FALSE
    file_suffix <- paste0("_rep_ge", min_sig_per_outcome)
    title_text <- sprintf(
      "Cluster %d (representative): outcomes with ≥%d significant cells",
      cluster_id,
      min_sig_per_outcome
    )
  } else {
    cluster_rows <- as.dendrogram(hc_row_sub)
    cluster_columns <- as.dendrogram(hc_col_sub)
    file_suffix <- ""
    title_text <- sprintf(
      "Cluster %d: %d exposures × %d outcomes",
      cluster_id,
      nrow(M_for_heat),
      ncol(M_for_heat)
    )
  }
  
  if (wrap_colnames) {
    colnames(M_for_heat) <- str_wrap(colnames(M_for_heat), width = wrap_width)
  }
  
  ht_star <- Heatmap(
    M_for_heat,
    name = "S",
    col = col_fun,
    cluster_rows = cluster_rows,
    cluster_columns = cluster_columns,
    show_row_names = TRUE,
    show_column_names = TRUE,
    column_title = title_text,
    column_names_rot = 45,
    row_names_gp = gpar(fontsize = 9),
    column_names_gp = gpar(fontsize = 9),
    row_names_max_width = unit(6, "cm"),
    border = "black",
    cell_fun = function(j, i, x, y, w, h, fill) {
      if (abs(M_for_heat[i, j]) >= sig_thr) {
        grid.text(
          "*",
          x,
          y = y - unit(star_offset, "snpc"),
          gp = gpar(fontsize = 14, col = "black"),
          just = "centre"
        )
      }
    }
  )
  
  pdf(
    sprintf("RowCluster%d_exposure_outcome_heatmap_STAR%s.pdf", cluster_id, file_suffix),
    width = width,
    height = height
  )
  
  draw(
    ht_star,
    heatmap_legend_side = "left",
    annotation_legend_side = "left",
    padding = padding,
    annotation_legend_list = list(lgd_star)
  )
  
  dev.off()
}

plot_row_cluster_heatmap(1, width = 8,  height = 7,    padding = unit(c(4, 1.5, 3, 5), "cm"), star_offset = 0.013)
plot_row_cluster_heatmap(2, width = 14, height = 8,    padding = unit(c(3, 2.5, 1.5, 1.5), "cm"), wrap_colnames = TRUE, star_offset = 0.013)
plot_row_cluster_heatmap(3, width = 14, height = 7.5,  padding = unit(c(3, 3, 1.5, 6), "cm"), star_offset = 0.005)
plot_row_cluster_heatmap(4, width = 18, height = 14.5, padding = unit(c(3, 3, 1.5, 7), "cm"), star_offset = 0.005)
plot_row_cluster_heatmap(5, width = 20, height = 8,    padding = unit(c(3, 3, 1.5, 3), "cm"), star_offset = 0.01)
plot_row_cluster_heatmap(6, width = 9,  height = 6,    padding = unit(c(3, 2, 1.5, 3), "cm"), star_offset = 0.01)
plot_row_cluster_heatmap(7, width = 7,  height = 5,    padding = unit(c(3, 1, 1.5, 4), "cm"), star_offset = 0.01)
plot_row_cluster_heatmap(8, width = 10, height = 9,    padding = unit(c(3, 1, 1.5, 5), "cm"), star_offset = 0.01)

plot_row_cluster_heatmap(
  2,
  width = 10.5,
  height = 5,
  padding = unit(c(3, 2.5, 1.5, 5), "cm"),
  representative = TRUE,
  min_sig_per_outcome = 8,
  wrap_colnames = TRUE,
  star_offset = 0.013
)

plot_row_cluster_heatmap(
  5,
  width = 10.5,
  height = 9.2,
  padding = unit(c(3, 2, 1.5, 5), "cm"),
  representative = TRUE,
  min_sig_per_outcome = 8,
  wrap_colnames = TRUE,
  star_offset = 0.013
)

############################################################
# Column-wise clustering, disease-driven clusters
############################################################

image_order <- c(
  "Cortical Surface Area",
  "Cortical Thickness",
  "Cortical Volume",
  "Subcortical Volume",
  "White Matter FA",
  "White Matter MD",
  "Functional Network Nodes",
  "Functional Connectivity Edges"
)

M_col <- M_all

brain_meta$exposuregroup <- factor(brain_meta$exposuregroup, levels = image_order)

exposure_order_col <- brain_meta %>%
  filter(exposure %in% keep_exposures) %>%
  arrange(factor(exposuregroup, levels = image_order)) %>%
  pull(exposure)

M_col <- M_col[exposure_order_col, , drop = FALSE]

brain_meta2 <- brain_meta[match(rownames(M_col), brain_meta$exposure), , drop = FALSE]
dis_meta2   <- dis_meta[match(colnames(M_col), dis_meta$Outcome), , drop = FALSE]

d_col <- as.dist(1 - cor(M_col, method = "pearson"))
hc_col <- hclust(d_col, method = "ward.D2")

############################################################
# Select number of column clusters
############################################################

p1_col <- fviz_nbclust(
  t(M_col),
  hcut,
  method = "wss",
  hc_method = "ward.D2",
  hc_metric = "pearson"
) +
  labs(
    title = "Elbow Method (Outcome clustering)",
    x = "Number of clusters (k)",
    y = "Total within-cluster sum of squares"
  ) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 14, color = "black"),
    axis.title.y = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 14, color = "black"),
    axis.text.y = element_text(size = 14, color = "black")
  )

sil_width_col <- numeric(max(k_candidates))

for (k in k_candidates) {
  cluster_k <- cutree(hc_col, k = k)
  sil <- silhouette(cluster_k, d_col)
  sil_width_col[k] <- mean(sil[, 3])
}

df_col <- data.frame(
  k = k_candidates,
  silhouette = sil_width_col[k_candidates]
)

p2_col <- ggplot(df_col, aes(x = k, y = silhouette)) +
  geom_line(color = "#5181B1") +
  geom_point(color = "#5181B1") +
  scale_x_continuous(breaks = k_candidates) +
  labs(
    title = "Silhouette Method (Outcome clustering)",
    x = "Number of clusters (k)",
    y = "Average silhouette width"
  ) +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 14, color = "black"),
    axis.title.y = element_text(size = 14, color = "black"),
    axis.text.x = element_text(size = 14, color = "black"),
    axis.text.y = element_text(size = 14, color = "black")
  )

ggsave(
  filename = "Outcome_Elbow_Silhouette.pdf",
  plot = p2_col + p1_col,
  width = 10,
  height = 5
)

# Final k was selected based on elbow plots, silhouette analysis,
# preservation of association patterns, and biological interpretability.
optimal_k_col <- 4
k_col <- optimal_k_col

col_clusters <- cutree(hc_col, k = k_col)

col_cluster_df <- data.frame(
  Outcome = names(col_clusters),
  col_cluster = factor(col_clusters)
) %>%
  left_join(dis_meta, by = "Outcome")

write.csv(col_cluster_df, "col_clusters_outcome.csv", row.names = FALSE)

############################################################
# Column-cluster heatmap
############################################################

row_anno2 <- rowAnnotation(
  ImageClass = brain_meta2$exposuregroup,
  col = list(ImageClass = ImageClass_colors)
)

cluster_colors <- c(
  brewer.pal(12, "Paired"),
  brewer.pal(8, "Set2")
)[1:k_col]

col_anno2 <- HeatmapAnnotation(
  DiseaseClass = dis_meta2$Outcomegroup,
  ColCluster = factor(col_clusters),
  col = list(
    DiseaseClass = diseaseclass_colors,
    ColCluster = structure(
      cluster_colors,
      names = levels(factor(col_clusters))
    )
  )
)

pdf("Heatmap_Bonferroni_ColCluster_sig_star.pdf", 11, 12)
draw(
  Heatmap(
    M_col,
    name = "S",
    col = col_fun,
    cluster_rows = FALSE,
    cluster_columns = as.dendrogram(hc_col),
    show_row_names = FALSE,
    show_column_names = FALSE,
    right_annotation = row_anno2,
    top_annotation = col_anno2,
    border = "black",
    cell_fun = function(j, i, x, y, w, h, fill) {
      if (abs(M_col[i, j]) >= sig_thr) {
        grid.text(
          "*",
          x = x,
          y = y - unit(0.005, "snpc"),
          gp = gpar(fontsize = 9, col = "black"),
          just = "centre"
        )
      }
    }
  ),
  annotation_legend_list = list(lgd_star)
)
dev.off()
