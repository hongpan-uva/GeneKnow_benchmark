library(ggplot2)
library(ggrepel)
library(cowplot)
library(scales)
library(dplyr)
library(tidyr)
library(ggsignif)
library(extrafont)
font_import("/mnt/d/projects/RAG_project/fonts", prompt = FALSE)

plot_dir = "/mnt/d/projects/RAG_project/plots/"
eval_dir = "/mnt/d/projects/RAG_project/evaluation/discover_cases/"
cases_included = paste0("case", seq(1,20))

methods_test = c("geneknow" = "GeneKnow Discover", "chatgpt5.2_thinking" = "ChatGPT 5.2 Thinking", "claude_opus4.6" = "Claude Opus 4.6", "gemini3_thinking"="Gemini3 Thinking")

## Read in data
parse_alignment_df <- function(df){
    out_vec = c("total" = NA, "supported"=NA, "opposed"=NA, "unsure"=NA, "non-verifiable"=NA)

    out_vec["non-verifiable"] <- length(which(df$paper_answer==""))
    df <- df[which(df$paper_answer!=""),]
    compare_vec <- df$claim_answer == df$paper_answer
    
    out_vec["total"] <- nrow(df)
    out_vec["supported"] <- length(which(compare_vec==TRUE))
    out_vec["unsure"] <- length(which(df$paper_answer=="Idk"))
    out_vec["opposed"] <- length(which(compare_vec==FALSE)) - out_vec["unsure"]

    return(out_vec)
}

parse_reference_df <- function(df){
    if(!all(df$real_reference %in% c("true", "false", "")) | !all(df$real_url %in% c("true", "false", ""))){
        stop("WRONG VALUE! not one of 'true', 'false' or ''")
    }

    out_vec = c("real paper&url" = NA, "fake paper"=NA, "fake url"=NA)

    n_fake_paper = length(which(df$real_reference == "false"))
    n_real_paper = length(which(df$real_reference == "true"))
    n_double_real = length(which(df$real_reference == "true" & df$real_url == "true"))
    n_fake_url = n_real_paper - n_double_real
    
    out_vec["real paper&url"] <- n_double_real
    out_vec["fake paper"] <- n_fake_paper
    out_vec["fake url"] <- n_fake_url

    return(out_vec)
}

alg_count_df <- data.frame()
ref_count_df <- data.frame()

for(case in cases_included){
    for(m in names(methods_test)){
        print(paste(case, m))
        alg_df <- read.csv(file.path(eval_dir, case, "claims", paste0(m, "_claims.csv")), header = T)
        alg_count_df <- rbind(alg_count_df, c("case" = case, "method" = m, parse_alignment_df(alg_df)))

        ref_df <- read.csv(file.path(eval_dir, case, "references", paste0(m, "_references.csv")), header = T)
        ref_count_df <- rbind(ref_count_df, c("case" = case, "method" = m, parse_reference_df(ref_df)))
    }
}

colnames(alg_count_df) <- c("case", "method", "total", "supported", "opposed", "unsure", "non-verifiable")
alg_count_df[c("total", "supported", "opposed", "unsure", "non-verifiable")] <- lapply(alg_count_df[c("total", "supported", "opposed", "unsure", "non-verifiable")], as.numeric)

colnames(ref_count_df) <- c("case", "method", "real paper&url", "fake paper", "fake url")
ref_count_df[c("real paper&url", "fake paper", "fake url")] <- lapply(ref_count_df[c("real paper&url", "fake paper", "fake url")], as.numeric)

alg_total_df <- alg_count_df %>%
    group_by(method) %>%
    summarise(
        across(c("supported", "opposed", "unsure", "non-verifiable"), sum),
        .groups = "drop"
    )

alg_total_long_df <- alg_total_df %>%
    pivot_longer(
        cols = c("supported", "opposed", "unsure", "non-verifiable"),
        names_to = "type",
        values_to = "count"
    )

ref_total_df <- ref_count_df %>%
    group_by(method) %>%
    summarise(
        across(c("real paper&url", "fake paper", "fake url"), sum),
        .groups = "drop"
    )

ref_total_long_df <- ref_total_df %>%
    pivot_longer(
        cols = c("real paper&url", "fake paper", "fake url"),
        names_to = "type",
        values_to = "count"
    )


# Plot claim stacked bar plot
# claim_type_colors = c("supported" = "#64a860", "unsure" = "#b98d3e", "opposed" = "#cc545e", "non-verifiable" = "grey50")
# claim_type_colors = c("supported" = "#a7d6ca", "unsure" = "#d2d1ae", "opposed" = "#e2b9c6", "non-verifiable" = "grey70")
claim_type_colors = c("supported" = "#b3cde3", "unsure" = "#fed9a6", "opposed" = "#fbb4ae", "non-verifiable" = "#e5d8bd")
alg_total_long_df$type <- factor(alg_total_long_df$type, levels = rev(c("supported", "unsure", "opposed", "non-verifiable")))
alg_total_long_df$method <- factor(alg_total_long_df$method, levels = c(names(methods_test)))

p1 <- ggplot(alg_total_long_df, aes(x = method, y = count, fill = type)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_x_discrete(labels = methods_test) +
    scale_fill_manual(values = claim_type_colors) +
    theme_bw() +
    labs(
        y = "Number of claims",
        fill = "claim type"
    ) +
    theme(
        aspect.ratio = 0.7,
        legend.position="none",
        text = element_text(size = 8, family = "Arial"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 6, color = "black", angle = 30, hjust = 0.9, vjust = 0.9),
        axis.text.y = element_text(size = 6, color = "black")
    )

ggsave(
    filename = "discover_claims_eval_stack_bar.png",
    plot = p1,
    device = "png",
    path = paste0(plot_dir, "discover"),
    width = 2.0,
    height = 2.2,
    dpi = 600,
)


# Claim pie plot
p_list <- list()
for(m in names(methods_test)){
    pie_df <- alg_total_long_df[which(alg_total_long_df$method == m),]
    pie_df <- pie_df[order(pie_df$type,decreasing = TRUE), ] # order by type factor levels, this is for correct percentage label

    pct_fun <- label_percent(accuracy = 0.01, scale = 100/sum(pie_df$count))
    label_df <- pie_df[which(pie_df$count > 0),] # df for percentage display
    label_df$pct <- sapply(label_df$count, pct_fun)
    label_df$pct_value <- as.numeric(gsub("%$", "", label_df$pct))
    label_df$cum_count <- cumsum(label_df$count)
    
    label_df_small <- label_df[which(label_df$pct_value < 5),]
    label_df_big <- label_df[which(label_df$pct_value >= 5 ),]

    p_list[[m]] <- ggplot(pie_df, aes(x="", y=count, fill=type)) +
        geom_bar(width = 1, stat = "identity") +
        scale_fill_manual(values = claim_type_colors) +
        coord_polar("y", start=0) +
        geom_text(data = label_df_big, mapping = aes(y = cum_count - count / 2, label = pct), size=1.5) +
        geom_text_repel( data = label_df_small, aes(y = cum_count - count / 2, label = pct), nudge_x = 1, direction = "y", segment.size = 0.2, size = 1.5) +
        theme_bw()+
        theme(
            legend.position="none",
            text = element_text(size = 6, family = "Arial"),
            plot.title = element_text(hjust = 0.5),
            plot.margin = unit(c(0,0,0,0), "in"),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.border = element_blank(),
            panel.grid=element_blank(),
            axis.ticks = element_blank(),
        )
}

set_null_device(cairo_pdf)
p_claim_pie <- plot_grid(p_list[[1]], p_list[[2]], p_list[[3]], p_list[[4]], nrow = 2, align = "vh")

for(m in names(methods_test)){
    ggsave(
        filename = paste0("discover_claims_pie_", m, ".png"),
        plot = p_list[[m]],
        device = "png",
        path = paste0(plot_dir, "discover_single_panels"),
        width = 1,
        height = 1,
        dpi = 600,
    )
}




# Plot reference stacked bar plot
ref_type_colors = c("real paper&url" = "#8dd3c7", "fake url" = "#fdb462", "fake paper" = "#fb8072")
ref_total_long_df$type <- factor(ref_total_long_df$type, levels = rev(c("real paper&url", "fake url", "fake paper")))
ref_total_long_df$method <- factor(ref_total_long_df$method, levels = c(names(methods_test)))

p3 <- ggplot(ref_total_long_df, aes(x = method, y = count, fill = type)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_x_discrete(labels = methods_test) +
    scale_fill_manual(values = ref_type_colors) +
    theme_bw() +
    labs(
        y = "Number of claims",
        fill = "claim type"
    ) +
    theme(
        aspect.ratio = 0.6,
        legend.position="none",
        text = element_text(size = 8, family = "Arial"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 6, color = "black", angle = 30, hjust = 0.9, vjust = 0.9),
        axis.text.y = element_text(size = 6, color = "black")
    )

ggsave(
    filename = "discover_reference_stack_bar.png",
    plot = p3,
    device = "png",
    path = paste0(plot_dir, "discover"),
    width = 2.0,
    height = 2.2,
    dpi = 600,
)

# Claim pie plot
p_list2 <- list()
for(m in names(methods_test)){
    pie_df <- ref_total_long_df[which(ref_total_long_df$method == m),]
    pie_df <- pie_df[order(pie_df$type,decreasing = TRUE), ] # order by type factor levels, this is for correct percentage label

    pct_fun <- label_percent(accuracy = 0.01, scale = 100/sum(pie_df$count))
    label_df <- pie_df[which(pie_df$count > 0),] # df for percentage display
    label_df$pct <- sapply(label_df$count, pct_fun)
    label_df$pct_value <- as.numeric(gsub("%$", "", label_df$pct))
    label_df$cum_count <- cumsum(label_df$count)
    
    label_df_small <- label_df[which(label_df$pct_value < 7),]
    label_df_big <- label_df[which(label_df$pct_value >= 7),]

    p_list2[[m]] <- ggplot(pie_df, aes(x="", y=count, fill=type)) +
        geom_bar(width = 1, stat = "identity") +
        scale_fill_manual(values = ref_type_colors) +
        coord_polar("y", start=0) +
        geom_text(data = label_df_big, mapping = aes(y = cum_count - count / 2, label = pct), size=2) +
        geom_text_repel(data = label_df_small, aes(y = cum_count - count / 2, label = pct), nudge_x = 1, direction = "y", segment.size = 0.2, size = 2) +
        # ggtitle(methods_test[m]) +
        theme_bw()+
        theme(
            legend.position="none",
            text = element_text(size = 6, family = "Arial"),
            plot.title = element_text(hjust = 0.5),
            plot.margin = unit(c(0,0,0,0), "in"),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.border = element_blank(),
            panel.grid=element_blank(),
            axis.ticks = element_blank(),
        )
}

set_null_device(cairo_pdf)
p_reference_pie <- plot_grid(p_list2[[1]], p_list2[[2]], p_list2[[3]], p_list2[[4]], nrow = 2, align = "vh")

for(m in names(methods_test)){
    ggsave(
        filename = paste0("discover_reference_pie_", m, ".png"),
        plot = p_list2[[m]],
        device = "png",
        path = paste0(plot_dir, "discover_single_panels"),
        width = 1,
        height = 1,
        dpi = 600,
    )
}

