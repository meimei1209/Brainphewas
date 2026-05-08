############################################################
# Category-level enrichment analysis
############################################################

############################################################
# Packages
############################################################

library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(circlize)
library(stringr)

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

dat_sig <- dat %>%
  filter(p < bonf_thr)

############################################################
# Category-level enrichment analysis
############################################################

dat_all_pairs <- dat %>%
  distinct(exposure, Outcome, exposuregroup, Outcomegroup)

dat_sig_pairs <- dat_sig %>%
  distinct(exposure, Outcome, exposuregroup, Outcomegroup)

M <- nrow(dat_all_pairs)
K <- nrow(dat_sig_pairs)

all_combinations <- expand.grid(
  exposuregroup = sort(unique(dat_all_pairs$exposuregroup)),
  Outcomegroup  = sort(unique(dat_all_pairs$Outcomegroup)),
  stringsAsFactors = FALSE
)

enrich_res <- all_combinations %>%
  rowwise() %>%
  mutate(
    N_pairs = sum(
      dat_all_pairs$exposuregroup == exposuregroup &
        dat_all_pairs$Outcomegroup == Outcomegroup
    ),
    k_sig = sum(
      dat_sig_pairs$exposuregroup == exposuregroup &
        dat_sig_pairs$Outcomegroup == Outcomegroup
    )
  ) %>%
  ungroup() %>%
  mutate(
    expected_sig = N_pairs * K / M,
    enrichment_ratio = ifelse(
      expected_sig > 0,
      k_sig / expected_sig,
      NA_real_
    ),
    p_hyper = phyper(
      q = k_sig - 1,
      m = K,
      n = M - K,
      k = N_pairs,
      lower.tail = FALSE
    ),
    p_Bonf = p.adjust(p_hyper, method = "bonferroni"),
    sig_label = ifelse(p_Bonf < 0.05, "*", "")
  )

write.csv(
  enrich_res,
  "enrichment_analysis_exposuregroup_outcomegroup.csv",
  row.names = FALSE
)

############################################################
# Heatmap for enrichment analysis
############################################################

exposure_order <- c(
  "Cortical Thickness",
  "Cortical Surface Area",
  "Cortical Volume",
  "Subcortical Volume",
  "White Matter FA",
  "White Matter MD",
  "Functional Network Nodes",
  "Functional Connectivity Edges"
)

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

enrich_res <- enrich_res %>%
  mutate(
    exposuregroup = factor(exposuregroup, levels = exposure_order),
    Outcomegroup  = factor(Outcomegroup, levels = outcome_order),
    enrichment_ratio_cap = enrichment_ratio
  )

p_enrich <- ggplot(
  enrich_res,
  aes(x = Outcomegroup, y = exposuregroup, fill = enrichment_ratio_cap)
) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sig_label), size = 7, vjust = 0.7) +
  scale_y_discrete(limits = rev) +
  scale_fill_gradient(
    low = "#F7FBFF",
    high = "#0072B2",
    name = "Enrichment ratio"
  ) +
  labs(
    x = "Disease category",
    y = "IDP category"
  ) +
  annotate(
    "text",
    x = Inf,
    y = 8,
    label = "* Statistically significant\n    enrichment",
    hjust = -0.1,
    vjust = 1,
    size = 4
  ) +
  coord_cartesian(clip = "off") +
  theme_bw() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      size = 11,
      family = "Arial",
      color = "black"
    ),
    axis.text.y = element_text(
      size = 11,
      family = "Arial",
      color = "black"
    ),
    axis.title.x = element_text(
      size = 13,
      family = "Arial",
      color = "black"
    ),
    axis.title.y = element_text(
      size = 13,
      family = "Arial",
      color = "black"
    ),
    legend.title = element_text(
      size = 11,
      family = "Arial",
      color = "black"
    ),
    legend.text = element_text(
      size = 10,
      family = "Arial",
      color = "black"
    ),
    panel.grid = element_blank(),
    plot.margin = margin(t = 10, r = 100, b = 10, l = 10)
  )

ggsave(
  filename = "enrichment_heatmap_ratio.pdf",
  plot = p_enrich,
  width = 11,
  height = 5,
  device = cairo_pdf,
  dpi = 300,
  fallback_resolution = 300
)

############################################################
# Flipped heatmap for enrichment analysis
############################################################

p_enrich_flip <- ggplot(
  enrich_res,
  aes(x = exposuregroup, y = Outcomegroup, fill = enrichment_ratio_cap)
) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sig_label), size = 7, vjust = 0.8) +
  scale_y_discrete(limits = rev) +
  scale_fill_gradient(
    low = "#F7FBFF",
    high = "#0072B2",
    name = "Enrichment ratio"
  ) +
  labs(
    x = "IDP category",
    y = "Disease category"
  ) +
  annotate(
    "text",
    x = 8,
    y = 12.5,
    label = "* Significant\n  enrichment",
    hjust = -0.4,
    vjust = 1,
    size = 4
  ) +
  coord_cartesian(clip = "off") +
  theme_bw() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      size = 11,
      family = "Arial",
      color = "black"
    ),
    axis.text.y = element_text(
      size = 11,
      family = "Arial",
      color = "black"
    ),
    axis.title.x = element_text(
      size = 13,
      family = "Arial",
      color = "black"
    ),
    axis.title.y = element_text(
      size = 13,
      family = "Arial",
      color = "black"
    ),
    legend.title = element_text(
      size = 11,
      family = "Arial",
      color = "black"
    ),
    legend.text = element_text(
      size = 10,
      family = "Arial",
      color = "black"
    ),
    panel.grid = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )

ggsave(
  filename = "enrichment_heatmap_ratio_flip.pdf",
  plot = p_enrich_flip,
  width = 6,
  height = 6.5,
  device = cairo_pdf,
  dpi = 300,
  fallback_resolution = 300
)

############################################################
# Chord plot for enrichment analysis
############################################################

chord_df <- enrich_res %>%
  filter(p_Bonf < 0.05, enrichment_ratio > 0) %>%
  mutate(
    exposuregroup = as.character(exposuregroup),
    Outcomegroup  = as.character(Outcomegroup),
    weight = enrichment_ratio
  ) %>%
  select(exposuregroup, Outcomegroup, enrichment_ratio, k_sig, p_Bonf, weight)

if (nrow(chord_df) == 0) {
  stop("No category pairs with p_Bonf < 0.05 for chord diagram.")
}

used_exposure <- exposure_order[exposure_order %in% unique(chord_df$exposuregroup)]
used_outcome  <- outcome_order[outcome_order %in% unique(chord_df$Outcomegroup)]

sector_order <- c(used_exposure, used_outcome)

grid_col <- c(
  "Cortical Thickness"             = "#D55E00",
  "Cortical Surface Area"          = "#0072B2",
  "Cortical Volume"                = "#CC79A7",
  "Subcortical Volume"             = "#F0E442",
  "White Matter FA"                = "#009E73",
  "White Matter MD"                = "#3182BD",
  "Functional Network Nodes"       = "#56B4E9",
  "Functional Connectivity Edges"  = "#999999",
  "Circulatory"                    = "#8DD3C7",
  "Dermatologic"                   = "#FFFFB3",
  "Digestive"                      = "#BEBADA",
  "Endocrine and metabolic"        = "#FB8072",
  "Genitourinary"                  = "#80B1D3",
  "Hematologic and immune"         = "#FFED6F",
  "Infectious"                     = "#B3DE69",
  "Musculoskeletal"                = "#FCCDE5",
  "Neoplasms"                      = "#D9D9D9",
  "Nervous"                        = "#BC80BD",
  "Ophthalmic and ENT"             = "#CCEBC5",
  "Psychiatric"                    = "#FDB462",
  "Respiratory"                    = "#1F78B4",
  "Symptomatic"                    = "#33A02C",
  "Traumatic and toxic"            = "#E31A1C"
)

link_col <- grid_col[chord_df$exposuregroup]

gap_degree <- c(
  rep(4, length(used_exposure) - 1),
  18,
  rep(2, length(used_outcome) - 1),
  18
)

stopifnot(length(sector_order) == length(gap_degree))

link_lwd <- 1 + 2 * log2(chord_df$enrichment_ratio + 1)

label_map <- c(
  "Cortical Thickness" = "Cortical\nThickness",
  "Cortical Surface Area" = "Cortical\nSurface Area",
  "Cortical Volume" = "Cortical\nVolume",
  "Subcortical Volume" = "Subcortical\nVolume",
  "White Matter FA" = "White Matter\nFA",
  "White Matter MD" = "White Matter\nMD",
  "Functional Network Nodes" = "Functional\nNetwork Nodes",
  "Functional Connectivity Edges" = "Functional Connectivity\nEdges",
  "Endocrine and metabolic" = "Endocrine\nand metabolic",
  "Hematologic and immune" = "Hematologic\nand immune",
  "Ophthalmic and ENT" = "Ophthalmic\nand ENT",
  "Traumatic and toxic" = "Traumatic\nand toxic"
)

pdf("chord_diagram_enrichment.pdf", width = 8.5, height = 8.5)
par(mar = c(1, 1, 1, 1))
circos.clear()
circos.par(
  start.degree = 100,
  gap.degree = gap_degree,
  track.margin = c(0.01, 0.01),
  cell.padding = c(0, 0, 0, 0),
  points.overflow.warning = FALSE,
  clock.wise = FALSE,
  canvas.xlim = c(-1.2, 1.2),
  canvas.ylim = c(-1.2, 1.2)
)

chordDiagram(
  x = chord_df[, c("exposuregroup", "Outcomegroup", "weight")],
  order = sector_order,
  grid.col = grid_col[sector_order],
  col = link_col,
  transparency = 0.25,
  directional = 1,
  direction.type = "arrows",
  link.arr.type = "big.arrow",
  diffHeight = mm_h(2),
  link.sort = TRUE,
  link.decreasing = FALSE,
  link.lwd = link_lwd,
  link.border = NA,
  annotationTrack = "grid",
  preAllocateTracks = list(track.height = 0.2)
)

circos.trackPlotRegion(
  track.index = 1,
  bg.border = NA,
  panel.fun = function(x, y) {
    sector_name <- get.cell.meta.data("sector.index")
    xcenter <- get.cell.meta.data("xcenter")
    ylim <- get.cell.meta.data("ylim")
    
    label_to_plot <- ifelse(
      sector_name %in% names(label_map),
      label_map[sector_name],
      sector_name
    )
    
    circos.text(
      x = xcenter,
      y = mean(ylim) - 0.3,
      labels = label_to_plot,
      facing = "clockwise",
      niceFacing = TRUE,
      adj = c(0.1, 0.5),
      cex = 1.4
    )
  }
)

circos.clear()
dev.off()