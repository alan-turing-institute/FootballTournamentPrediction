# load packages
library(tidyverse)
library(gt)
library(gtExtras)
library(webshot2)

# before World Cup
original <- read.csv("original_predictions.csv")
# after Round 1 games
round_1 <- read.csv("after_round_1.csv")
# after Round 1 games and Wales-Iran
round_1_WI <- read.csv("after_round_1_wales_iran.csv")
# after Round 1 games (updated)
round_1_updated <- read.csv("after_round_1_updated.csv")
# after Round 2 games
round_2 <- read.csv("after_round_2.csv")
# after Round 3 games (predicting R16 onwards)
R16 <- read.csv("R16.csv")
# after R16 (predicting QF onwards)
QF <- read.csv("QF.csv")
# after QF (predicting SF onwards)
SF <- read.csv("SF.csv")
# after SF (predicting F)
Final <- read.csv("F.csv")

get_progression_probabiltiies <- function(df, n_sim, from_round) {
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
    mutate('logo' = paste0('../flags/', Team, '.png')) %>%
    select(Team, logo, everything()) %>%
    gt() %>%
    tab_header(
      title = md("**Men's World Cup 2022**"),
      subtitle = subtitle) %>% 
    cols_label(Team = "",
               logo = "") %>%
    cols_width(Team ~ 80, 
               logo ~ 50,
               everything() ~ 60) %>% 
    fmt_percent(columns = (2:ncol(data_frame))+1, 
                decimals = 1) %>% 
    data_color(columns = (2:ncol(data_frame))+1,
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
                                 mutate('logo' = paste0('../flags/', Team, '.png')) %>%
                                 select(Team, logo, everything()))
  if (!is.null(filename)) {
    gtsave(full_table, filename = filename)  
  }
  return(full_table)
}

# original plot
original_prob <- get_progression_probabiltiies(original, 100000)

create_table_plot(original_prob, c(-0.5, 1), filename = "plots/predictions.png")
create_table_plot(original_prob[1:10,], c(-0.5, 1), filename = "plots/predictions_top_10.png")

# after round 1 plots
after_round_1_prob <- get_progression_probabiltiies(round_1, 100000)
after_round_1_diff <- get_progression_prob_differences(new_df = after_round_1_prob,
                                                       old_df = original_prob)

create_table_plot(data_frame = after_round_1_prob,
                  domain = c(-0.5, 1),
                  subtitle = "After Round 1",
                  filename = "plots/after_round_1.png")
create_table_plot(data_frame = after_round_1_diff,
                  domain = c(-0.5, 0.5),
                  subtitle = "Difference in probability after Round 1",
                  filename = "plots/after_round_1_diff.png")
create_table_plot_alt_colours(data_frame = after_round_1_prob,
                              data_frame_for_colours = after_round_1_diff,
                              domain = c(-0.5, 0.5),
                              subtitle = "After Round 1",
                              filename = "plots/after_round_1_colour_diff.png")

# after round 1 plots with Wales-Iran
after_round_1_prob <- get_progression_probabiltiies(round_1_WI, 100000)
after_round_1_diff <- get_progression_prob_differences(new_df = after_round_1_prob,
                                                       old_df = original_prob)

create_table_plot(data_frame = after_round_1_prob,
                  domain = c(-0.5, 1),
                  subtitle = "After Round 1 (and Wales-Iran)",
                  filename = "plots/after_round_1_WI.png")
create_table_plot(data_frame = after_round_1_diff,
                  domain = c(-0.5, 0.5),
                  subtitle = "Difference in probability after Round 1 (and Wales-Iran)",
                  filename = "plots/after_round_1_WI_diff.png")
create_table_plot_alt_colours(data_frame = after_round_1_prob,
                              data_frame_for_colours = after_round_1_diff,
                              domain = c(-0.5, 0.5),
                              subtitle = "After Round 1 (and Wales-Iran)",
                              filename = "plots/after_round_1_WI_colour_diff.png")

# after round 1 plots (updated)
after_round_1_prob <- get_progression_probabiltiies(round_1_updated, 100000)
after_round_1_diff <- get_progression_prob_differences(new_df = after_round_1_prob,
                                                       old_df = original_prob)

create_table_plot(data_frame = after_round_1_prob,
                  domain = c(-0.5, 1),
                  subtitle = "After Round 1",
                  filename = "plots/after_round_1_updated.png")
create_table_plot(data_frame = after_round_1_diff,
                  domain = c(-0.5, 0.5),
                  subtitle = "Difference in probability after Round 1",
                  filename = "plots/after_round_1_updated_diff.png")
create_table_plot_alt_colours(data_frame = after_round_1_prob,
                              data_frame_for_colours = after_round_1_diff,
                              domain = c(-0.5, 0.5),
                              subtitle = "After Round 1",
                              filename = "plots/after_round_1_updated_colour_diff.png")


# after round 2 plots
after_round_2_prob <- get_progression_probabiltiies(round_2, 100000)
after_round_2_diff <- get_progression_prob_differences(new_df = after_round_2_prob,
                                                       old_df = after_round_1_prob)

create_table_plot(data_frame = after_round_2_prob,
                  domain = c(-0.5, 1),
                  subtitle = "After Round 2",
                  filename = "plots/after_round_2.png")
create_table_plot(data_frame = after_round_2_diff,
                  domain = c(-0.5, 0.5),
                  subtitle = "Difference in probability between Round 2 and Round 1",
                  filename = "plots/after_round_2_dff.png")
create_table_plot_alt_colours(data_frame = after_round_2_prob,
                              data_frame_for_colours = after_round_2_diff,
                              domain = c(-0.5, 0.5),
                              subtitle = "After Round 2",
                              filename = "plots/after_round_2_colour_diff.png")

# after round 3 plots
R16_prob <- get_progression_probabiltiies(R16, 100000)

create_table_plot(data_frame = R16_prob[1:16, c("Team", "QF", "SF", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Round of 16",
                  filename = "plots/R16.png")

# after R16 plots
QF_prob <- get_progression_probabiltiies(QF, 100000)

create_table_plot(data_frame = QF_prob[1:8, c("Team", "SF", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Quarter Finalists",
                  filename = "plots/QF.png")

# after SF plots
SF_prob <- get_progression_probabiltiies(SF, 100000)

create_table_plot(data_frame = SF_prob[1:4, c("Team", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Semi Finalists",
                  filename = "plots/SF.png")

# after F plots
F_prob <- get_progression_probabiltiies(Final, 100000)

create_table_plot(data_frame = F_prob[1:2, c("Team", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Finalists",
                  filename = "plots/F.png")
 