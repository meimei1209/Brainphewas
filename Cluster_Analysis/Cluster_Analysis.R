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

############################################################
# Load PheWAS summary results and data preparation
############################################################
# dat contains:
#   exposure  – IDP name
#   Case_N    – number of incident cases
#   Control_N – number of controls
#   Person_years
#   HR, CI95, p
#   phenotype (phecode)
#   Outcome (disease)
#   Outcomegroup (disease category)

# Keep only outcomes with Case_N ≥ 20
dat <- dat %>%
  filter(Case_N >= 20)

# Remove congenital and pregnancy-related diseases
dat <- dat %>% filter(Outcomegroup != "Congenital")
dat <- dat %>% filter(Outcomegroup != "Pregnancy")

# Replace fMRI node/edge names using reference file
fMRIinfo <- read.csv("fMRI_node_edge_information_R.csv")
names(fMRIinfo)
dat$exposure <- ifelse(
  dat$exposure %in% fMRIinfo$IDP,
  fMRIinfo$IDP_Network[match(dat$exposure, fMRIinfo$IDP)],
  dat$exposure
)

# Add imaging modality category for each exposure
dat <- dat %>%
  mutate(
    exposuregroup = case_when(
      str_starts(exposure, "Mean")   ~ "Cortical Thickness",
      str_starts(exposure, "Area")   ~ "Cortical Surface Area",
      str_starts(exposure, "Volume") ~ "Cortical Volume",
      str_starts(exposure, "MD")     ~ "White Matter MD",
      str_starts(exposure, "FA")     ~ "White Matter FA",
      str_starts(exposure, "Node")  ~ "Functional Network Nodes",
      str_starts(exposure, "Edge")  ~ "Functional Connectivity Edges",
      TRUE                           ~ "Subcortical Volume"
    ),
    exposure = if_else(exposuregroup == "Subcortical Volume",
                       paste0("Volume of ", exposure),
                       exposure)
  )

# Compute Bonferroni threshold based on all IDP × outcome tests
n_exp  <- n_distinct(dat$exposure)
n_out  <- n_distinct(dat$Outcome)
n_test <- n_exp * n_out
bonf_thr <- 0.05 / n_test
sig_thr <- -log10(bonf_thr)

dat_sig <- dat %>% filter(p < bonf_thr)
keep_exposures <- unique(dat_sig$exposure)
keep_outcomes  <- unique(dat_sig$Outcome)

############################################################
# Build signed -log10(P) matrix S for significant pairs
############################################################

# Brain and disease meta-data
brain_meta <- dat %>%
  distinct(exposure, exposuregroup)
dis_meta <- dat %>%
  distinct(Outcome, Outcomegroup)

# Compute S = sign(beta) * -log10(p)
dat2 <- dat %>%
  mutate(
    beta    = log(HR),
    sign    = if_else(beta >= 0, 1, -1),
    mlog10p = -log10(p),
    S       = sign * mlog10p
  )

# Wide matrix: rows = exposures, columns = outcomes
M_all <- dat2 %>%
  filter(exposure %in% keep_exposures,
         Outcome  %in% keep_outcomes) %>%
  select(exposure, Outcome, S) %>%
  pivot_wider(names_from = Outcome, values_from = S) %>%
  as.data.frame()

rownames(M_all) <- M_all$exposure
M_all$exposure  <- NULL

############################################################
# Color schemes
############################################################

ImageClass_colors <- c(
  "Cortical Surface Area"         = "#E41A1C",
  "Cortical Thickness"            = "#377EB8",
  "Cortical Volume"               = "#999999",
  "Functional Connectivity Edges" = "#984EA3",
  "Functional Network Nodes"      = "#FF7F00",
  "Subcortical Volume"            = "#A65628",
  "White Matter FA"               = "#F781BF",
  "White Matter MD"               = "#1B9E77"
)

diseaseclass_colors <- c(
  "Circulatory"             = "#1B9E77",
  "Dermatologic"            = "#7570B3",
  "Digestive"               = "#1F78B4",
  "Endocrine and metabolic" = "#66A61E",
  "Genitourinary"           = "#E6AB02",
  "Hematologic and immune"  = "#A6761D",
  "Infectious"              = "#666666",
  "Musculoskeletal"         = "#E7298A",
  "Neoplasms"               = "#B2DF8A",
  "Nervous"                 = "#FB9A99",
  "Ophthalmic and ENT"      = "#A6CEE3",
  "Psychiatric"             = "#CAB2D6",
  "Respiratory"             = "#FF7F00",
  "Symptomatic"             = "#6A3D9A",
  "Traumatic and toxic"     = "#B15928"
)

# Color scale for S
col_fun <- colorRamp2(
  c(-10, 0, 10),
  c("#3b4cc0", "white", "#b40426")
)

############################################################
# 1. Row-wise clustering (brain-driven clusters)
############################################################

group_order <- c(
  "Neoplasms","Endocrine and metabolic","Hematologic and immune",
  "Psychiatric","Nervous","Ophthalmic and ENT","Circulatory",
  "Respiratory","Digestive","Genitourinary","Dermatologic",
  "Musculoskeletal","Symptomatic", "Traumatic and toxic"
)

M_row <- M_all

dis_meta$Outcomegroup <- factor(dis_meta$Outcomegroup, levels = group_order)

outcome_order <- dis_meta %>%
  filter(Outcome %in% colnames(M_row)) %>%
  arrange(factor(Outcomegroup, levels = group_order)) %>%
  pull(Outcome)

M_row <- M_row[, outcome_order, drop = FALSE]

brain_meta2 <- brain_meta[match(rownames(M_row), brain_meta$exposure), , drop = FALSE]
dis_meta2   <- dis_meta[match(colnames(M_row), dis_meta$Outcome), , drop = FALSE]

# Distance and clustering
d_row <- as.dist(1 - cor(t(M_row), method = "pearson"))
hc_row <- hclust(d_row, method = "ward.D2")

k_row <- 10
row_clusters <- cutree(hc_row, k = k_row)

row_cluster_df <- data.frame(
  exposure = names(row_clusters),
  row_cluster = factor(row_clusters)
) %>% 
  left_join(brain_meta, by = "exposure")
write.csv(row_cluster_df, "row_clusters_exposure.csv", row.names = FALSE)

# Row-cluster heatmap with significance stars
row_cluster_colors <- c(
  brewer.pal(12, "Paired"),
  brewer.pal(8, "Set2")
)[1:k_row]

row_anno2 <- rowAnnotation(
  ImageClass = brain_meta2$exposuregroup,
  RowCluster = factor(row_clusters),
  col = list(
    ImageClass = ImageClass_colors,
    RowCluster = structure(row_cluster_colors, 
                           names = levels(factor(row_clusters)))
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
    grid.text("*",
              x, y,
              gp = gpar(fontsize = 12, col = "black", fontface = "bold"))
  })
)

pdf("Heatmap_Bonferroni_RowCluster_sig_star.pdf",12,9)
draw(
  Heatmap(M_row, name="S", 
          col=col_fun,
          cluster_rows=as.dendrogram(hc_row), 
          cluster_columns=FALSE,
          show_row_names=FALSE, 
          show_column_names=FALSE,
          right_annotation=row_anno2, 
          top_annotation=col_anno2,
          cell_fun = function(j, i, x, y, w, h, fill) {
            if(abs(M_row[i, j]) >= sig_thr) {
              grid.text("*", 
                        x = x, 
                        y = y - unit(0.0045, "snpc"),  
                        gp = gpar(fontsize = 10, col = "black"),
                        just = "centre")
            }
          }
  ),
  annotation_legend_list = list(lgd_star)   
)
dev.off()

############################################################
# 1.1 Sub-clustering within a selected brain-driven cluster
############################################################

cluster_id <- 1  # <- change this to 1–k_row as needed

# Exposures in the selected cluster
exp_in_cl <- row_cluster_df %>%
  filter(row_cluster == cluster_id) %>%
  pull(exposure)

# Outcomes with Bonferroni-significant associations with these exposures
dis_in_cl <- dat2 %>%
  filter(exposure %in% exp_in_cl, p < bonf_thr) %>%
  pull(Outcome) %>%
  unique()

# Build S matrix for this subcluster 
subM_df <- dat2 %>%
  filter(exposure %in% exp_in_cl, Outcome %in% dis_in_cl) %>%
  select(exposure, Outcome, S) %>%
  pivot_wider(names_from = Outcome, values_from = S) %>%
  as.data.frame()

rownames(subM_df) <- subM_df$exposure
subM_df$exposure  <- NULL

subM_df[] <- lapply(subM_df, function(x) suppressWarnings(as.numeric(x)))
M_for_heat <- as.matrix(subM_df)
storage.mode(M_for_heat) <- "double"
M_for_heat <- M_for_heat[exp_in_cl, dis_in_cl, drop = FALSE]

# Clustering based on S
d_row <- as.dist(1 - cor(t(M_for_heat), method = "pearson", use = "pairwise.complete.obs"))
d_col <- as.dist(1 - cor(M_for_heat,    method = "pearson", use = "pairwise.complete.obs"))
hc_row <- hclust(d_row, method = "ward.D2")
hc_col <- hclust(d_col, method = "ward.D2")

# Heatmap with stars
ht_star <- Heatmap(
  M_for_heat,
  name = "S",
  col = col_fun,
  cluster_rows = as.dendrogram(hc_row),
  cluster_columns = as.dendrogram(hc_col),
  show_row_names = TRUE,
  show_column_names = TRUE,
  column_title = sprintf("Cluster %d: %d exposures × %d outcomes", 
                         cluster_id, nrow(M_for_heat), ncol(M_for_heat)),
  column_names_rot = 45,
  row_names_gp = gpar(fontsize = 9),
  column_names_gp = gpar(fontsize = 9),
  row_names_max_width = unit(6, "cm"),
  border = "black",
  
  cell_fun = function(j, i, x, y, w, h, fill) {
    if (abs(M_for_heat[i, j]) >= sig_thr) {
      grid.text("*", x, y = y - unit(0.013, "snpc"),
                gp = gpar(fontsize = 14, col = "black"),
                just = "centre")
    }
  }
)

lgd_star <- Legend(
  labels = "Significant",
  type = "graphics",
  graphics = list(function(x, y, w, h) {
    grid.text("*", x, y,
              gp = gpar(fontsize = 12, col = "black", fontface = "bold"))
  })
)

pdf(sprintf("RowCluster%d_exposure_outcome_heatmap_STAR.pdf", cluster_id), width = 10, height = 7)  
draw(
  ht_star,
  heatmap_legend_side = "left",
  annotation_legend_side = "left",
  padding = unit(c(4, 5, 3, 5), "cm"), # top, right, bottom, left
  annotation_legend_list = list(lgd_star) 
)
dev.off()

############################################################
# 2. Column-wise clustering (disease-driven clusters)
############################################################

image_order <- c(
  "Cortical Surface Area","Cortical Thickness","Cortical Volume",
  "Subcortical Volume","White Matter FA","White Matter MD",
  "Functional Network Nodes","Functional Connectivity Edges"
)

M_col <- M_all

brain_meta$exposuregroup <- factor(brain_meta$exposuregroup, levels=image_order)

exposure_order <- brain_meta %>%
  filter(exposure %in% keep_exposures) %>%  
  arrange(factor(exposuregroup, levels=image_order)) %>%
  pull(exposure)

M_col <- M_col[exposure_order, , drop=FALSE]  

brain_meta2 <- brain_meta[match(rownames(M_col), brain_meta$exposure), , drop = FALSE]
dis_meta2   <- dis_meta[match(colnames(M_col), dis_meta$Outcome), , drop = FALSE]

d_col <- as.dist(1 - cor(M_col, method="pearson"))
hc_col <- hclust(d_col, method="ward.D2")

k_col <- 10
col_clusters <- cutree(hc_col, k=k_col)

col_cluster_df <- data.frame(
  Outcome = names(col_clusters),
  col_cluster = factor(col_clusters)
) %>% 
  left_join(dis_meta, by="Outcome")
write.csv(col_cluster_df, "col_clusters_outcome.csv", row.names=FALSE)

# Column-cluster heatmap with significance stars
row_anno2 <- rowAnnotation(
  ImageClass=brain_meta2$exposuregroup,
  col=list(ImageClass=ImageClass_colors)
)

cluster_colors <- c(brewer.pal(12, "Paired"),brewer.pal(8, "Set2"))[1:k_col]

col_anno2 <- HeatmapAnnotation(
  DiseaseClass = dis_meta2$Outcomegroup,
  ColCluster   = factor(col_clusters),
  col = list(
    DiseaseClass = diseaseclass_colors,
    ColCluster   = structure(cluster_colors, names = levels(factor(col_clusters)))
  )
)

lgd_star <- Legend(
  labels = "Significant",
  type = "graphics",
  graphics = list(function(x, y, w, h) {
    grid.text("*", x, y,
              gp = gpar(fontsize = 12, col = "black", fontface = "bold"))
  })
)

pdf("Heatmap_Bonferroni_ColCluster_sig_star.pdf", 11, 9)
draw(Heatmap(M_col, name="S", col=col_fun,
             cluster_rows=FALSE, cluster_columns=as.dendrogram(hc_col),
             show_row_names=FALSE, show_column_names=FALSE,
             right_annotation=row_anno2, top_annotation=col_anno2,
             border="black",
             cell_fun = function(j, i, x, y, w, h, fill) {
               if (abs(M_col[i, j]) >= sig_thr) {
                 grid.text("*", 
                           x = x, 
                           y = y - unit(0.005, "snpc"),  
                           gp = gpar(fontsize = 9, col = "black"),
                           just = "centre")
               }
             }),
     annotation_legend_list=list(lgd_star))
dev.off()
