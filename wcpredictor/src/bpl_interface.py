"""
Interface to the NumPyro team model in bpl-next:
https://github.com/anguswilliams91/bpl-next
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bpl import NeutralDixonColesMatchPredictor, NeutralDixonColesMatchPredictorWC
from bpl.base import BaseMatchPredictor

from wcpredictor.src.data_loader import get_confederations_data

WC_HOSTS = {
    "2002": "South Korea",  # and Japan
    "2006": "Germany",
    "2010": "South Africa",
    "2014": "Brazil",
    "2018": "Russia",
    "2022": "Qatar",
}


class WCPred:
    def __init__(
        self,
        results: pd.DataFrame,
        fixtures: Optional[pd.DataFrame] = None,
        ratings: Optional[pd.DataFrame] = None,
        teams: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        epsilon: float = 0.0,
        world_cup_weight: float = 1.0,
        weights_dict: Optional[dict[str, float]] = None,
        model: BaseMatchPredictor = None,
        host: str = "Qatar",
    ):
        self.results = results
        self.fixtures = fixtures
        self.ratings = ratings
        if teams is None:
            # teams is just every team that has played in results
            self.teams = list(
                set(self.results["home_team"]) | set(self.results["away_team"])
            )
        else:
            # disregard any games which involves a team that isnt in teams
            self.teams = teams
            self.results = self.results[
                self.results.home_team.isin(teams) & self.results.away_team.isin(teams)
            ]
        if years is not None:
            self.results = self.results[self.results.date.dt.year.isin(years)]
        confed = get_confederations_data()
        self.confed_dict = dict(zip(confed["Team"], confed["Confederation"]))
        self.training_data = None
        self.epsilon = epsilon
        self.world_cup_weight = world_cup_weight
        self.weights_dict = weights_dict
        self.model = model
        self.host = host

    def get_result_dict(self) -> dict[str, np.array]:
        """
        put results into dictionary to train model
        """
        return {
            "home_team": np.array(self.results.home_team),
            "away_team": np.array(self.results.away_team),
            "home_conf": np.array(
                [self.confed_dict[team] for team in self.results.home_team]
            ),
            "away_conf": np.array(
                [self.confed_dict[team] for team in self.results.away_team]
            ),
            "home_goals": np.array(self.results.home_score),
            "away_goals": np.array(self.results.away_score),
            "neutral_venue": np.array(self.results.neutral),
            "time_diff": np.array(self.results.time_diff),
            "game_weights": np.array(self.results.game_weight),
        }

    def get_ratings_dict(self) -> dict:
        """Create a dataframe containing the fifa team ratings."""
        ratings = self.ratings[self.ratings.Team.isin(self.teams)]
        ratings_dict = {
            row.Team: np.array(row.drop("Team").values.astype(float))
            for _, row in ratings.iterrows()
        }
        if len(ratings_dict) != len(self.teams):
            raise ValueError(
                f"Must have FIFA ratings and results for all teams. {len(ratings_dict)}"
                + f" teams with FIFA ratings but {len(self.teams)} teams with results."
            )
        return ratings_dict

    def check_teams_in_ratings(self) -> bool:
        """Validate whether there are (e.g. FIFA) ratings for all the teams"""
        if self.ratings is None:
            return True
        teams = pd.Series(self.teams)
        print("---len(teams[~teams.isin(self.ratings.Team)])")
        if len(teams[~teams.isin(self.ratings.Team)]) > 0:
            raise ValueError(
                "Must have FIFA ratings and results for all teams. There are "
                + f"{len(teams[~teams.isin(self.ratings.Team)])} "
                + "teams with no FIFA ratings:\n\n"
                + ", ".join(teams[~teams.isin(self.ratings.Team)])
            )
        return True

    def set_training_data(self) -> None:
        """Get training data for team model including FIFA ratings as covariates.
        Data returned is for all matches up to specified gameweek and season.
        """
        print("[MODEL FITTING] Setting training data for the model")
        training_data = self.get_result_dict()
        if self.ratings is not None:
            if self.ratings.isna().any().any():
                raise ValueError(
                    "There are some NaN values in ratings, please fix or remove"
                )
            if self.check_teams_in_ratings:
                training_data["team_covariates"] = self.get_ratings_dict()
        self.training_data = training_data

    def fit_model(self, model=None, **fit_args) -> None:
        """
        Get the team-level stan model, which can give probabilities of
        each potential scoreline in a given fixture.
        """
        if model is not None:
            self.model = model
        if self.model is None:
            self.model = NeutralDixonColesMatchPredictorWC(max_goals=10)
        if self.training_data is None:
            self.set_training_data()
        print("[MODEL FITTING] Fitting the model")
        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            if not fit_args:
                fit_args = {}
            fit_args["epsilon"] = self.epsilon

        self.model = self.model.fit(self.training_data, **fit_args)

    def get_fixture_probabilities(
        self,
        home_team: Union[str, List[str]],
        away_team: Union[str, List[str]],
        knockout: bool = False,
    ) -> dict:
        """
        Returns probabilities and predictions for the fixtures defined by home_team,
        and away_team as a dict with keys home_team, away_team, home_win, away_win,
        and draw, where the last 3 are the probabilities of that result. If knockout
        is True don't consider the possibility of draws.
        """
        (
            home_team,
            away_team,
            home_conference,
            away_conference,
            venue,
        ) = self._parse_sim_args(home_team, away_team)

        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            p = self.model.predict_outcome_proba(
                home_team, away_team, home_conference, away_conference, venue, knockout
            )
        elif isinstance(self.model, NeutralDixonColesMatchPredictor):
            p = self.model.predict_outcome_proba(home_team, away_team, venue, knockout)
        else:
            p = self.model.predict_outcome_proba(home_team, away_team, knockout)

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "home_win": p["home_win"],
            "away_win": p["away_win"],
        }
        if not knockout:
            result["draw"] = p["draw"]
        return result

    def _parse_sim_args(
        self, home_team: Union[str, List[str]], away_team: Union[str, List[str]]
    ):
        if isinstance(home_team, str):
            home_team = [home_team]
        if isinstance(away_team, str):
            away_team = [away_team]
        home_team = np.array(home_team)
        away_team = np.array(away_team)
        home_conference = np.array([self.confed_dict[team] for team in home_team])
        away_conference = np.array([self.confed_dict[team] for team in away_team])

        # ensure host nation always the home team
        venue = np.ones(len(home_team))
        away_team_host = away_team == self.host
        away_team[away_team_host] = home_team[away_team_host]
        away_conference[away_team_host] = home_conference[away_team_host]
        home_team[away_team_host] = self.host
        home_conference[away_team_host] = self.confed_dict[self.host]
        venue[home_team == self.host] = 0

        return home_team, away_team, home_conference, away_conference, venue

    def sample_score(
        self,
        home_team: Union[str, List[str]],
        away_team: Union[str, List[str]],
        num_samples: int = 1,
        seed: int = None,
    ) -> dict:
        """Sample the score of matches between home_team and away_team num_samples
        times, returned as a dict with keys home_team, away_team, home_score, away_score
        """
        (
            home_team,
            away_team,
            home_conference,
            away_conference,
            venue,
        ) = self._parse_sim_args(home_team, away_team)

        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            result = self.model.sample_score(
                home_team,
                away_team,
                home_conference,
                away_conference,
                venue,
                num_samples,
                seed,
            )

        elif isinstance(self.model, NeutralDixonColesMatchPredictor):
            result = self.model.sample_score(
                home_team,
                away_team,
                venue,
                num_samples,
                seed,
            )

        else:
            result = self.model.sample_score(
                home_team,
                away_team,
                num_samples,
                seed,
            )
        result["home_team"] = home_team
        result["away_team"] = away_team
        return result

    def sample_outcome(
        self,
        home_team: Union[str, List[str]],
        away_team: Union[str, List[str]],
        knockout: bool = False,
        num_samples: float = 1,
        seed: int = None,
    ) -> np.ndarray:
        """Sample the outcome of matches between home_team and away_team num_samples
        times, returned as an array of strings representing the winning team or 'Draw'
        ('Draw' only considered if knockout is False)
        """
        (
            home_team,
            away_team,
            home_conference,
            away_conference,
            venue,
        ) = self._parse_sim_args(home_team, away_team)

        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            return self.model.sample_outcome(
                home_team,
                away_team,
                home_conference,
                away_conference,
                venue,
                knockout,
                num_samples,
                seed,
            )

        elif isinstance(self.model, NeutralDixonColesMatchPredictor):
            return self.model.sample_outcome(
                home_team,
                away_team,
                venue,
                knockout,
                num_samples,
                seed,
            )

        else:
            return self.model.sample_outcome(
                home_team,
                away_team,
                knockout,
                num_samples,
                seed,
            )

    def get_fixture_score_probabilities(
        self, home_team: Union[str, List[str]], away_team: Union[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the proability of exact scorelines in fixtures between home_team
        and away_team. Returned values are (probabilities, home_score, away_score)"""
        (
            home_team,
            away_team,
            home_conference,
            away_conference,
            venue,
        ) = self._parse_sim_args(home_team, away_team)

        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            return self.model.predict_score_grid_proba(
                home_team, away_team, home_conference, away_conference, venue
            )
        elif isinstance(self.model, NeutralDixonColesMatchPredictor):
            return self.model.predict_score_grid_proba(home_team, away_team, venue)
        else:
            return self.model.predict_score_grid_proba(home_team, away_team)

    def get_fixture_team_goal_probabilities(
        self,
        home_team: Union[str, List[str]],
        away_team: Union[str, List[str]],
        max_goals: int = 10,
    ) -> dict:
        """
        Get the probability that each team in a fixture scores any number of goals up to
        max_goals, and prediction of goals scored. Returned as dict with keys goals,
        home_team, away_team, home_prob, away_prob.
        """
        (
            home_team,
            away_team,
            home_conference,
            away_conference,
            venue,
        ) = self._parse_sim_args(home_team, away_team)
        goals = np.arange(0, max_goals + 1)

        if isinstance(self.model, NeutralDixonColesMatchPredictorWC):
            home_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                home_team,
                away_team,
                home_conference,
                away_conference,
                home=True,
                neutral_venue=venue,
            )
            away_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                away_team,
                home_team,
                away_conference,
                home_conference,
                home=False,
                neutral_venue=venue,
            )
        elif isinstance(self.model, NeutralDixonColesMatchPredictor):
            home_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                home_team,
                away_team,
                home=True,
                neutral_venue=venue,
            )
            away_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                away_team,
                home_team,
                home=False,
                neutral_venue=venue,
            )
        else:
            home_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                home_team,
                away_team,
                home=True,
            )
            away_team_goal_prob = self.model.predict_score_n_proba(
                goals,
                away_team,
                home_team,
                home=False,
            )

        return {
            "goals": goals,
            "home_team": home_team,
            "away_team": away_team,
            "home_prob": home_team_goal_prob,
            "away_prob": away_team_goal_prob,
        }

    def get_most_probable_scoreline(
        self, home_team: Union[str, List[str]], away_team: Union[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most probable scoreline for matches between home_team and
        away_team. Returned values are arrays home_goals, away_goals, and the
        probability of that scoreline"""
        probs, home_goals, away_goals = self.get_fixture_score_probabilities(
            home_team, away_team
        )
        probs = probs.reshape((len(home_team), -1))
        home_goals = home_goals.flatten()
        away_goals = away_goals.flatten()
        idx = np.argmax(probs, axis=-1)
        return home_goals[idx], away_goals[idx], probs[:, idx]
