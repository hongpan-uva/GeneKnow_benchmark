library(ggplot2)
library(cowplot)
library(dplyr)
library(tidyr)
library(ggsignif)
library(extrafont)
font_import("/mnt/d/projects/RAG_project/fonts", prompt = FALSE)

plot_dir = "/mnt/d/projects/RAG_project/plots"
eval_dir = "/mnt/d/projects/RAG_project/evaluation/inspect_cases"
cases_included = paste0("case", seq(1,20))

methods_test = c("GeneKnow_inspect" = "GeneKnow Inspect", "naive_inspect_1" = "Full-paper LLM", "naive_inspect_2" = "Hierarchical LLM")

## Read in data
parse_alignment_df <- function(df){
    out_vec = c("total" = NA, "supported"=NA, "opposed"=NA, "unsure"=NA)

    df <- df[which(df$evidence_answer!=""),]
    compare_vec <- df$claim_answer == df$evidence_answer
    
    out_vec["total"] <- nrow(df)
    out_vec["supported"] <- length(which(compare_vec==TRUE))
    out_vec["unsure"] <- length(which(df$evidence_answer=="Idk"))
    out_vec["opposed"] <- length(which(compare_vec==FALSE)) - out_vec["unsure"]

    return(out_vec)
}

parse_coverage_df <- function(df){
    out_vec = c("total" = NA, "supported"=NA, "opposed"=NA, "unsure"=NA)

    df <- df[which(!is.na(df$claim_answer)),]
    compare_vec <- df$claim_answer == df$summary_answer
    
    out_vec["total"] <- nrow(df)
    out_vec["supported"] <- length(which(compare_vec==TRUE))
    out_vec["unsure"] <- length(which(df$summary_answer=="Idk"))
    out_vec["opposed"] <- length(which(compare_vec==FALSE)) - out_vec["unsure"]

    return(out_vec)
}

alg_count_df <- data.frame()
cvg_count_df <- data.frame()

for(case in cases_included){
    for(m in names(methods_test)){
        alg_df <- read.csv(file.path(eval_dir, case, "alignment_eval", paste0(m, "_claims.csv")), header = T)
        alg_count_df <- rbind(alg_count_df, c("case" = case, "method" = m, parse_alignment_df(alg_df)))
        
        cvg_df <- read.csv(file.path(eval_dir, case, "coverage_eval", paste0(m, "_coverage.csv")), header = T)
        cvg_count_df <- rbind(cvg_count_df, c("case" = case, "method" = m, parse_coverage_df(cvg_df)))
    }
}

colnames(alg_count_df) <- c("case", "method", "total", "supported", "opposed", "unsure")
alg_count_df[c("total", "supported", "opposed", "unsure")] <- lapply(alg_count_df[c("total", "supported", "opposed", "unsure")], as.numeric)
colnames(cvg_count_df) <- c("case", "method", "total", "supported", "opposed", "unsure")
cvg_count_df[c("total", "supported", "opposed", "unsure")] <- lapply(cvg_count_df[c("total", "supported", "opposed", "unsure")], as.numeric)

stat_df <- data.frame(
    case = alg_count_df$case,
    method = alg_count_df$method,
    alignment = alg_count_df$supported / alg_count_df$total,
    coverage = cvg_count_df$supported / cvg_count_df$total
)

stat_df$f1 <- 2 * stat_df$alignment * stat_df$coverage / (stat_df$alignment + stat_df$coverage)
stat_df$method <- factor(stat_df$method, levels = c("GeneKnow_inspect", "naive_inspect_2", "naive_inspect_1"))

alg_total_df <- alg_count_df %>%
    group_by(method) %>%
    summarise(
        across(c(total, supported, opposed, unsure), sum),
        .groups = "drop"
    )

alg_total_long_df <- alg_total_df %>%
    pivot_longer(
        cols = c(supported, opposed, unsure),
        names_to = "type",
        values_to = "count"
    )

# Define manual colors
color_palette <- c(
  "GeneKnow_inspect" = "#4F7A8E",
  "naive_inspect_1" = "#afcddb",
  "naive_inspect_2" = "#7eabbf"
)

boxplot_layers <- list(
    geom_boxplot(width = 0.7, outlier.shape = NA),
    geom_jitter(width = 0.2, size = 1, alpha = 0.6),
    stat_summary(
        fun = mean,geom = "point",
        shape = 23, size = 2, fill = "white", color = "black"
    ),
    scale_x_discrete(labels = methods_test),
    scale_fill_manual(values = color_palette),
    ylim(0.18, 1.02),
    theme_bw(),
    theme(
        # aspect.ratio = 0.6,
        legend.position="none",
        text = element_text(size = 8, family = "Arial"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(color = "black", size = 6, angle = 30, hjust = 0.9, vjust = 0.9),
        axis.text.y = element_text(color = "black", size = 6),
        plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "in")
    )
)

# Plot alignment score
p1 <- ggplot(stat_df, aes(x = method, y = alignment, fill = method)) +
    labs(y = "Alignment") +
    boxplot_layers

# Plot coverage score
p2 <- ggplot(stat_df, aes(x = method, y = coverage, fill = method)) +
    labs(y = "Coverage") +
    boxplot_layers

# Plot F1 score
p3 <- ggplot(stat_df, aes(x = method, y = f1, fill = method)) +
    labs(y = "F1 score") +
    boxplot_layers

# Plot claim stacked bar plot
claim_type_colors = c("supported" = "#b3cde3", "unsure" = "#fed9a6", "opposed" = "#fbb4ae", "non-verifiable" = "#e5d8bd")
alg_total_long_df$type <- factor(alg_total_long_df$type, levels = rev(c("supported", "unsure", "opposed")))
alg_total_long_df$method <- factor(alg_total_long_df$method, levels = c("GeneKnow_inspect", "naive_inspect_2", "naive_inspect_1"))

p4 <- ggplot(alg_total_long_df, aes(x = method, y = count, fill = type)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_x_discrete(labels = methods_test) +
    scale_fill_manual(values = claim_type_colors) +
    theme_bw() +
    labs(
        y = "Number of claims",
        fill = "claim type"
    ) +
    theme(
        # aspect.ratio = 0.6,
        legend.position="none",
        text = element_text(size = 8, family = "Arial"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(color = "black", size = 6, angle = 30, hjust = 0.9, vjust = 0.9),
        axis.text.y = element_text(color = "black", size = 6),
        plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "in")
    )

set_null_device(cairo_pdf)
p_joint <- plot_grid(p4, p1, p2, p3, nrow = 1, align = "h", rel_widths = c(1, 0.8, 0.8, 0.8))

ggsave(
    filename = "inspect_eval_stats.png",
    plot = p_joint,
    device = "png",
    path = plot_dir,
    width = 6.5,
    height = 2.0,
    dpi = 600,
)

