# load packages
library(tidyverse)
library(gt)
library(gtExtras)
library(webshot2)

# before World Cup
 original <- read.csv("original_predictions.csv")
# after Round 3 games (predicting R16 onwards)
R16 <- read.csv("R16.csv")
# after R16 (predicting QF onwards)
QF <- read.csv("QF.csv")
# after QF (predicting SF onwards)
SF <- read.csv("SF.csv")
# after SF first game
SF <- read.csv("SF_after_first_game.csv")
# # after SF (predicting F)
# Final <- read.csv("F.csv")

# probabilities of knock out at each round
cbind(original$Team, original[c("Group", "R16", "QF", "SF", "RU", "W")] / 100000)

get_progression_probabilties <- function(df, n_sim, from_round) {
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
  prog_difference <- data.frame(matrix(nrow=0, ncol=5))
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
      title = md("**Women's World Cup 2023**"),
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
      source_note = md("The Alan Turing Institute (Nick Barlow, Jack Roberts, Ryan Chan)<br>Based on 100,000 simulations<br>Codebase: GitHub (alan-turing-institute/WorldCupPrediction)<br>Data: GitHub (martj42/womens-international-results)<br>Country Images: Flaticon.com")
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
original_prob <- get_progression_probabilties(original, 100000)

create_table_plot(original_prob, c(-0.5, 1), filename = "plots/predictions.png")
create_table_plot(original_prob[1:10,], c(-0.5, 1), filename = "plots/predictions_top_10.png")

# after round 3 plots
R16_prob <- get_progression_probabilties(R16, 100000)

create_table_plot(data_frame = R16_prob[1:16, c("Team", "QF", "SF", "F", "W")],
                   domain = c(-0.5, 1),
                   subtitle = "Round of 16",
                   filename = "plots/R16.png")

# after R16 plots
QF_prob <- get_progression_probabilties(QF, 100000)

create_table_plot(data_frame = QF_prob[1:8, c("Team", "SF", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Quarter Finalists",
                  filename = "plots/QF.png")

# after SF plots
SF_prob <- get_progression_probabilties(SF, 100000)

create_table_plot(data_frame = SF_prob[1:4, c("Team", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Semi Finalists",
                  filename = "plots/SF.png")

# after SF plots (after first game)
SF_prob <- get_progression_probabilties(SF_after_first_game, 100000)

create_table_plot(data_frame = SF_prob[1:4, c("Team", "F", "W")],
                  domain = c(-0.5, 1),
                  subtitle = "Semi Finalists",
                  filename = "plots/SF.png")


# # after F plots
# F_prob <- get_progression_probabilties(Final, 100000)
#
# create_table_plot(data_frame = F_prob[1:2, c("Team", "W")],
#                   domain = c(-0.5, 1),
#                   subtitle = "Finalists",
#                   filename = "plots/F.png")
#