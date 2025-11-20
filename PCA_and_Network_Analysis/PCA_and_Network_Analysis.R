############################################################
# Packages
############################################################
library(dplyr)
library(tidyr)
library(ggplot2)
library(igraph)
library(tidygraph)
library(stringr)

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
M <- dat2 |>
  filter(exposure %in% keep_exposures, Outcome %in% keep_outcomes) |>
  select(exposure, Outcome, S) |>
  tidyr::pivot_wider(names_from = Outcome, values_from = S) |>
  as.data.frame()

rownames(M) <- M$exposure
M$exposure <- NULL

############################################################
# 1. Principal Component Analysis (PCA)
############################################################

pca <- prcomp(M, center = FALSE, scale. = FALSE)

scores <- as.data.frame(pca$x) %>%
  tibble::rownames_to_column("exposure") %>%
  left_join(brain_meta, by = "exposure")

p <- ggplot(scores, aes(PC1, PC2, color = exposuregroup)) +
  geom_point(alpha = 1, size = 2.5) +
  stat_ellipse(type = "norm", linetype = "solid", size = 2, alpha = 0.4) + 
  labs(
    x = sprintf("PC1 (%.1f%%)", 100 * summary(pca)$importance[2, 1]),
    y = sprintf("PC2 (%.1f%%)", 100 * summary(pca)$importance[2, 2])
  ) +
  theme_classic(base_size = 18) +  
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 20),   
    axis.title.x = element_text(size = 18),  
    axis.title.y = element_text(size = 18),  
    axis.text.x = element_text(size = 16, color = "black"), 
    axis.text.y = element_text(size = 16, color = "black"), 
    axis.line = element_line(color = "black", size = 1),    
    axis.ticks = element_line(color = "black", size = 1)
  )
ggsave("PCA_plot.pdf", plot = p, width = 10, height = 5.5)

############################################################
# 2. Network analysis
############################################################
#Build bipartite network edges
edges <- dat2 %>%
  filter(p <= bonf_thr) %>%
  transmute(
    from = paste0("B_", exposure),
    to   = paste0("D_", Outcome),
    w    = abs(S),
    dir  = as.character(sign),
    w_vis = sqrt(abs(S))
  )

g_bip <- graph_from_data_frame(edges, directed = FALSE)
V(g_bip)$type <- grepl("^B_", V(g_bip)$name)  

g_tbl <- as_tbl_graph(g_bip) %>%
  mutate(
    node_type = ifelse(type, "Imaging", "Disease"),
    exposure  = ifelse(type, sub("^B_", "", name), NA),
    outcome   = ifelse(!type, sub("^D_", "", name), NA),
    degree    = centrality_degree()
  )

node_info <- g_tbl %>%
  as_tibble() %>%
  select(name, node_type, degree, exposure, outcome)

# Node metadata table
node_info <- node_info %>%
  left_join(brain_meta, by = c("exposure" = "exposure")) %>%
  left_join(dis_meta,   by = c("outcome"  = "Outcome")) %>%
  mutate(
    Group = ifelse(node_type == "Imaging", exposuregroup, Outcomegroup),
    name  = sub("^[BD]_","", name)   # ✅ 去掉 B_ / D_
  ) %>%
  select(name, node_type, Group, degree)

write.csv(node_info, "node_degree_results_with_group.csv", row.names = FALSE)

# Significant edges for inspection
edges_sig <- dat2 %>%
  filter(p <= bonf_thr) %>%
  mutate(effect_size = abs(beta)) %>%
  select(exposure, Outcome, HR, beta, p, effect_size)

write.csv(edges_sig, "Edges_significant_effectsize.csv", row.names = FALSE)

############################################################
# Cytoscape-compatible tables
############################################################
edges_cyto <- dat2 %>%
  filter(p <= bonf_thr) %>%
  transmute(
    Source = exposure,
    Target = Outcome,
    Weight = abs(S),
    Sign   = sign,
    HR     = HR,
    beta   = beta,
    p      = p
  )

write.table(edges_cyto, "cytoscape_edges.tsv",
            sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

# Imaging nodes
nodes_imaging <- brain_meta %>%
  filter(exposure %in% unique(edges_cyto$Source)) %>%
  transmute(
    `shared name` = exposure,  
    Type  = "Imaging",
    Group = exposuregroup
  )

# Disease nodes
nodes_disease <- dis_meta %>%
  filter(Outcome %in% unique(edges_cyto$Target)) %>%
  transmute(
    `shared name` = Outcome,  
    Type  = "Disease",
    Group = Outcomegroup
  )

nodes_cyto <- bind_rows(nodes_imaging, nodes_disease)

# Label top-degree nodes for Cytoscape
top_imaging <- g_tbl %>%
  filter(node_type == "Imaging") %>%
  arrange(desc(degree)) %>%
  slice_head(n = 10) %>%
  mutate(name_clean = sub("^B_", "", name)) %>% 
  pull(name_clean)

top_disease <- g_tbl %>%
  filter(node_type == "Disease") %>%
  arrange(desc(degree)) %>%
  slice_head(n = 10) %>%
  mutate(name_clean = sub("^D_", "", name)) %>% 
  pull(name_clean)

label_nodes <- c(top_imaging, top_disease)

nodes_cyto <- nodes_cyto %>%
  mutate(Label = ifelse(`shared name` %in% label_nodes, `shared name`, ""))

write.table(nodes_cyto, "cytoscape_nodes.tsv",
            sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

