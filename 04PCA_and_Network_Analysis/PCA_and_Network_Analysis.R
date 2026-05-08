############################################################
# Packages
############################################################
library(dplyr)
library(tidyr)
library(ggplot2)
library(igraph)
library(tidygraph)
library(stringr)
library(forcats)
library(grid)
library(ggsci)

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
  filter(p < bonf_thr) %>%
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
    name  = sub("^[BD]_","", name)    # Remove B_ / D_ prefixes
  ) %>%
  select(name, node_type, Group, degree)

write.csv(node_info, "node_degree_results_with_group.csv", row.names = FALSE)

# Significant edges for inspection
edges_sig <- dat2 %>%
  filter(p < bonf_thr) %>%
  mutate(effect_size = abs(beta)) %>%
  select(exposure, Outcome, HR, beta, p, effect_size)

write.csv(edges_sig, "Edges_significant_effectsize.csv", row.names = FALSE)

############################################################
# Cytoscape-compatible tables
############################################################
edges_cyto <- dat2 %>%
  filter(p < bonf_thr) %>%
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

############################################################
# Hub node and edge barplots
############################################################

############################################################
# Top imaging nodes by degree
############################################################

node_degree_dat <- node_info %>%
  mutate(degree = as.numeric(degree))

top10_img <- node_degree_dat %>%
  filter(node_type == "Imaging") %>%
  arrange(desc(degree), name) %>%
  distinct(name, .keep_all = TRUE) %>%
  slice_head(n = 10)

p_img <- ggplot(
  top10_img,
  aes(x = degree, y = reorder(name, degree))
) +
  geom_col(width = 0.7, fill = "#6C94A8") +
  labs(
    x = "Number of significant connections",
    y = "IDP",
    title = "Top 10 IDPs by degree"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.y  = element_text(size = 12, color = "black"),
    axis.text.x  = element_text(size = 12, color = "black"),
    axis.title   = element_text(size = 12, color = "black"),
    plot.title   = element_text(hjust = 0, size = 12, color = "black")
  )

ggsave(
  filename = "top_imaging_nodes_by_degree.pdf",
  plot = p_img,
  width = 4,
  height = 5.0
)

############################################################
# Top disease nodes by degree
############################################################

top10_dis <- node_degree_dat %>%
  filter(node_type == "Disease") %>%
  arrange(desc(degree), name) %>%
  distinct(name, .keep_all = TRUE) %>%
  slice_head(n = 10) %>%
  mutate(name_wrapped = str_wrap(name, width = 40))

p_dis <- ggplot(
  top10_dis,
  aes(x = degree, y = reorder(name_wrapped, degree))
) +
  geom_col(width = 0.7, fill = "#6C94A8") +
  labs(
    x = "Number of significant connections",
    y = "Disease",
    title = "Top 10 diseases by degree"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x  = element_text(size = 12, color = "black", hjust = 1),
    axis.text.y  = element_text(size = 12, color = "black"),
    axis.title.x = element_text(size = 12, color = "black"),
    axis.title.y = element_text(size = 12, color = "black"),
    plot.title   = element_text(hjust = 0, size = 12, color = "black")
  )

ggsave(
  filename = "top_disease_nodes_by_degree.pdf",
  plot = p_dis,
  width = 5.6,
  height = 5.0
)

############################################################
# Top IDP-disease pairs by effect size
############################################################

edge_plot_dat <- edges_sig %>%
  mutate(
    p = as.numeric(p),
    beta = as.numeric(beta),
    effect_size = as.numeric(effect_size),
    pair = paste(exposure, Outcome, sep = " - "),
    logp = -log10(p)
  )

top10_eff <- edge_plot_dat %>%
  arrange(desc(effect_size), p) %>%
  slice_head(n = 10) %>%
  mutate(pair_wrapped = str_replace(pair, " - ", " -\n"))

p_eff <- ggplot(
  top10_eff,
  aes(x = effect_size, y = reorder(pair_wrapped, effect_size))
) +
  geom_col(width = 0.7, fill = "#6C94A8") +
  labs(
    x = "Effect size (|Beta|)",
    y = "IDP-disease pair",
    title = "Top 10 pairs by effect size"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x  = element_text(size = 12, color = "black", hjust = 1),
    axis.text.y  = element_text(size = 12, color = "black"),
    axis.title.x = element_text(size = 12, color = "black"),
    axis.title.y = element_text(size = 12, color = "black"),
    plot.title   = element_text(hjust = 0, size = 12, color = "black"),
    plot.margin  = margin(t = 5, r = 5, b = 5, l = 100, unit = "pt")
  )

ggsave(
  filename = "top_pairs_by_effect_size.pdf",
  plot = p_eff,
  width = 6.5,
  height = 5.1
)

############################################################
# Top IDP-disease pairs by statistical significance
############################################################

top10_p <- edge_plot_dat %>%
  arrange(p, desc(effect_size)) %>%
  slice_head(n = 10) %>%
  mutate(pair_wrapped = str_replace(pair, " - ", " -\n"))

p_sig <- ggplot(
  top10_p,
  aes(x = logp, y = reorder(pair_wrapped, logp))
) +
  geom_col(width = 0.7, fill = "#6C94A8") +
  labs(
    x = expression(-log[10](italic(P))),
    y = "IDP-disease pair",
    title = "Top 10 pairs by significance"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x  = element_text(size = 12, color = "black", hjust = 1),
    axis.text.y  = element_text(size = 12, color = "black"),
    axis.title.x = element_text(size = 12, color = "black"),
    axis.title.y = element_text(size = 12, color = "black"),
    plot.title   = element_text(hjust = 0, size = 12, color = "black"),
    plot.margin  = margin(t = 5, r = 5, b = 5, l = 70, unit = "pt")
  )

ggsave(
  filename = "top_pairs_by_significance.pdf",
  plot = p_sig,
  width = 6.3,
  height = 5.1
)

############################################################
# Cluster-level hub barplots
############################################################
# The following section requires cluster assignment tables generated
# from the clustering analysis script:
#   row_cluster_df: IDP cluster assignments, with columns exposure and row_cluster
#   col_cluster_df: disease cluster assignments, with columns Outcome and col_cluster
#
# These tables are used to identify the top-connected imaging and disease
# nodes within each detected IDP or disease cluster.
############################################################

############################################################
# Top imaging hubs within each IDP cluster
############################################################

node_info_clustered <- node_info %>%
  mutate(degree = as.numeric(degree)) %>%
  left_join(
    row_cluster_df %>%
      select(
        name = exposure,
        IDP_cluster = row_cluster
      ),
    by = "name"
  ) %>%
  left_join(
    col_cluster_df %>%
      select(
        name = Outcome,
        Disease_cluster = col_cluster
      ),
    by = "name"
  )

top_n_idp <- 3

plot_idp <- node_info_clustered %>%
  filter(
    node_type == "Imaging",
    !is.na(IDP_cluster)
  ) %>%
  group_by(IDP_cluster) %>%
  arrange(desc(degree), name, .by_group = TRUE) %>%
  slice_head(n = top_n_idp) %>%
  ungroup() %>%
  mutate(
    IDP_cluster = factor(
      IDP_cluster,
      levels = sort(unique(IDP_cluster)),
      labels = paste("IDP cluster", sort(unique(IDP_cluster)))
    ),
    name = str_wrap(name, width = 40)
  ) %>%
  group_by(IDP_cluster) %>%
  mutate(name = fct_reorder(name, degree, .desc = TRUE)) %>%
  ungroup()

p_idp_cluster <- ggplot(plot_idp, aes(x = name, y = degree, fill = Group)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = degree),
    vjust = -0.35,
    size = 3.8,
    family = "Arial",
    color = "black"
  ) +
  facet_wrap(~ IDP_cluster, scales = "free_x", nrow = 1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    x = NULL,
    y = "Number of significant\nconnections",
    fill = "Imaging class"
  ) +
  theme_bw(base_family = "Arial") +
  theme(
    strip.text = element_text(
      size = 11,
      face = "bold",
      color = "black",
      family = "Arial"
    ),
    axis.text.x = element_text(
      size = 10,
      angle = 45,
      hjust = 1,
      vjust = 1,
      color = "black",
      family = "Arial"
    ),
    axis.text.y = element_text(
      size = 11,
      color = "black",
      family = "Arial"
    ),
    axis.title.y = element_text(
      size = 12,
      color = "black",
      family = "Arial"
    ),
    legend.title = element_text(
      size = 11,
      color = "black",
      family = "Arial",
      vjust = 1.2
    ),
    legend.text = element_text(
      size = 10,
      color = "black",
      family = "Arial"
    ),
    legend.position = "right",
    legend.justification = c(0.5, 0.58),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.spacing = unit(0.25, "lines"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 75)
  ) +
  coord_cartesian(clip = "off") +
  scale_fill_nejm()

ggsave(
  filename = "idp_cluster_hub_barplot.pdf",
  plot = p_idp_cluster,
  width = 15,
  height = 5.0,
  device = cairo_pdf,
  dpi = 300,
  fallback_resolution = 300
)

############################################################
# Top disease hubs within each disease cluster
############################################################

top_n_dis <- 5

plot_dis_cluster <- node_info_clustered %>%
  filter(
    node_type == "Disease",
    !is.na(Disease_cluster)
  ) %>%
  group_by(Disease_cluster) %>%
  arrange(desc(degree), name, .by_group = TRUE) %>%
  slice_head(n = top_n_dis) %>%
  ungroup() %>%
  mutate(
    Disease_cluster = factor(
      Disease_cluster,
      levels = sort(unique(Disease_cluster)),
      labels = paste("Disease cluster", sort(unique(Disease_cluster)))
    ),
    name = str_wrap(name, width = 40)
  ) %>%
  group_by(Disease_cluster) %>%
  mutate(name = fct_reorder(name, degree, .desc = TRUE)) %>%
  ungroup()

p_disease_cluster <- ggplot(plot_dis_cluster, aes(x = name, y = degree, fill = Group)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = degree),
    vjust = -0.35,
    size = 3.8,
    family = "Arial",
    color = "black"
  ) +
  facet_wrap(~ Disease_cluster, scales = "free_x", nrow = 1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    x = NULL,
    y = "Number of significant\nconnections",
    fill = "Disease class"
  ) +
  theme_bw(base_family = "Arial") +
  theme(
    strip.text = element_text(
      size = 11,
      face = "bold",
      color = "black",
      family = "Arial"
    ),
    axis.text.x = element_text(
      size = 10,
      angle = 45,
      hjust = 1,
      vjust = 1,
      color = "black",
      family = "Arial"
    ),
    axis.text.y = element_text(
      size = 11,
      color = "black",
      family = "Arial"
    ),
    axis.title.y = element_text(
      size = 12,
      color = "black",
      family = "Arial"
    ),
    legend.title = element_text(
      size = 11,
      color = "black",
      family = "Arial",
      vjust = 1.2
    ),
    legend.text = element_text(
      size = 10,
      color = "black",
      family = "Arial"
    ),
    legend.position = "right",
    legend.justification = c(0.5, 0.58),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.spacing = unit(0.25, "lines"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 75)
  ) +
  coord_cartesian(clip = "off") +
  scale_fill_nejm()

ggsave(
  filename = "disease_cluster_hub_barplot.pdf",
  plot = p_disease_cluster,
  width = 14.7,
  height = 4.8,
  device = cairo_pdf,
  dpi = 300,
  fallback_resolution = 300
)