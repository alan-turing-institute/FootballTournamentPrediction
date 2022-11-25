# load packages
library(tidyverse)
library(gt)
library(gtExtras)
library(webshot2)

# # before World Cup
original <- read.csv("merged_1668598471_ep_2_wc_4_100000.csv")
# after Round 1 games
round_1 <- read.csv("1669331060_sim_results.csv")

get_progression_probabiltiies <- function(df, n_sim) {
  df <- df[order(df$W, decreasing = TRUE),]
  prog_df <- data.frame(matrix(nrow=0, ncol = 5))
  for (i in 1:nrow(df)) {
    progression <- c(sum(df[i, c("R16", "QF", "SF", "RU", "W")]) / n_sim,
                     sum(df[i, c("QF", "SF", "RU", "W")]) / n_sim,
                     sum(df[i, c("SF", "RU", "W")]) / n_sim,
                     sum(df[i, c("RU", "W")]) / n_sim,
                     sum(df[i, c("W")]) / n_sim)
    prog_df <- rbind(prog_df, progression)
  }
  if ("team" %in% colnames(df)) {
    prog_df <- cbind(df$team, prog_df)  
  } else if ("Team" %in% colnames(df)) {
    prog_df <- cbind(df$Team, prog_df)  
  }
  colnames(prog_df) <- c("Team", "R16", "QF", "SF", "F", "W")
  return(prog_df)
}

get_progression_prob_differences <- function(new_df, old_df) {
  prog_difference <- data.frame(matrix(nrow=0, ncol = 5))
  for (i in 1:nrow(new_df)) {
    team <- new_df[i,]$Team
    new_probabilities <- new_df[new_df$Team==team, c("R16", "QF", "SF", "F", "W")]
    old_probabilities <- old_df[old_df$Team==team, c("R16", "QF", "SF", "F", "W")]
    prog_difference <- rbind(prog_difference, new_probabilities-old_probabilities)
  }
  if ("team" %in% colnames(new_df)) {
    prog_difference <- cbind(new_df$team, prog_difference)  
  } else if ("Team" %in% colnames(new_df)) {
    prog_difference <- cbind(new_df$Team, prog_difference)  
  }
  colnames(prog_difference) <- c("Team", "R16", "QF", "SF", "F", "W")
  return(prog_difference)
}

create_table_plot <- function(data_frame,
                              domain,
                              subtitle = "",
                              filename = NULL) {
  full_table <- data_frame %>%
    mutate('logo' = paste0('flags/', Team, '.png')) %>%
    select(Team, logo, everything()) %>%
    gt() %>%
    tab_header(
      title = md("**World Cup 2022**"),
      subtitle = subtitle) %>% 
    cols_label(Team = "",
               logo = "") %>%
    cols_width(Team ~ 80, 
               logo ~ 50,
               everything() ~ 60) %>% 
    fmt_percent(columns = c(R16, QF, SF, F, W), 
                decimals = 1) %>% 
    data_color(columns = c(R16, QF, SF, F, W),
               colors = scales::col_numeric(
                 palette = paletteer::paletteer_d(
                   palette = "Redmonder::dPBIRdGn",
                   direction = 1
                 ) %>% as.character(),
                 domain = domain, 
                 na.color = "#005C55FF"
               )) %>%
    text_transform(
      locations = cells_body(columns = "logo"), 
      fn = function(x) map_chr(x, ~{
        local_image(filename =  as.character(.x), height = 30)
      })
    ) %>%
    tab_options(
      heading.title.font.size = 20,
      heading.subtitle.font.size = 14,
      heading.title.font.weight = 'bold',
      column_labels.font.size = 14,
      column_labels.font.weight = 'bold',
      table.font.size = 12,
      source_notes.font.size = 8,
      data_row.padding = px(.8)
    ) %>%
    tab_source_note(
      source_note = md("The Alan Turing Institute (Nick Barlow, Jack Roberts, Ryan Chan)<br>Based on 100,000 simulations<br>Data: GitHub (martj42/international_results)<br>Country Images: Flaticon.com and GitHub (lbenz730/world_cup_2022)")
    )
  if (!is.null(filename)) {
    gtsave(full_table, filename = filename)  
  }
  return(full_table)
}

create_table_plot_alt_colours <- function(data_frame,
                                          data_frame_for_colours,
                                          domain,
                                          subtitle = "",
                                          filename = NULL) {
  full_table <- create_table_plot(data_frame_for_colours, domain, subtitle)
  full_table$`_data` <- tibble(data_frame %>%
                                 mutate('logo' = paste0('flags/', Team, '.png')) %>%
                                 select(Team, logo, everything()))
  if (!is.null(filename)) {
    gtsave(full_table, filename = filename)  
  }
  return(full_table)
}

# original plot
original_prob <- get_progression_probabiltiies(original, 100000)

create_table_plot(original_prob, c(-0.5, 1), filename = "plots/predictions.png")

# after round 1 plots
after_round_1_prob <- get_progression_probabiltiies(round_1, 100000)
after_round_1_diff <- get_progression_prob_differences(new_df = after_round_1_prob,
                                                       old_df = original_prob)

create_table_plot(data_frame = prog_round_1,
                  domain = c(-0.5, 1),
                  subtitle = "After Round 1",
                  filename = "plots/after_round_1.png")
create_table_plot(data_frame = diff,
                  domain = c(-0.5, 0.5),
                  subtitle = "Difference in probability after Round 1",
                  filename = "plots/after_round_1_diff.png")
create_table_plot_alt_colours(data_frame = prog_round_1,
                              data_frame_for_colours = diff,
                              domain = c(-0.5, 0.5),
                              subtitle = "After Round 1",
                              filename = "plots/after_round_1_colour_diff.png")

 