"""
Code to run the World Cup tournament, from group stages through the
knockout stages, to the final, and produce a winner.
"""

import os
import pandas as pd
import random


def predict_knockout_match(team_1, team_2):
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    winning_team: str, one of team_1 or team_2
    """
    if random.random()>0.5:
        return team_2
    else:
        return team_1


def predict_group_match(team_1, team_2):
    """
    Parameters
    ==========
    team_1, team_2: both str, names of two teams

    Returns:
    ========
    score_1, score_2: both int, score for each team
    """
    if random.random()>0.2:
        return (1,0)
    elif random.random()>0.4:
        return (0,1)
    elif random.random()>0.6:
        return (1,1)
    elif random.random()>0.8:
        return (3,2)
    else:
        return (0,0)


class Group:
    def __init__(self, name, teams):
        self.name = name
        self.teams = teams
        self.table = {}
        for t in self.teams:
            self.table[t] = {"points":0, "goals_for": 0, "goals_against": 0}
        self.results = []

    def play_match(self, fixture):
        """
        Play a simulated group match.

        Parameters
        ==========
        fixture: row from a pandas DataFrame, has Team_1 and Team_2 columns

        Returns
        =======
        result: dict, where keys are the team names,
                and values are the goals for that team
        """
        goals_1, goals_2 = predict_group_match(fixture.Team_1, fixture.Team_2)
        self.table[fixture.Team_1]["goals_for"] += goals_1
        self.table[fixture.Team_2]["goals_for"] += goals_2
        self.table[fixture.Team_1]["goals_against"] += goals_2
        self.table[fixture.Team_2]["goals_against"] += goals_1
        if goals_1 > goals_2:
            self.table[fixture.Team_1]["points"] += 3
        elif goals_2 > goals_1:
            self.table[fixture.Team_2]["points"] += 3
        else:
            self.table[fixture.Team_1]["points"] += 1
            self.table[fixture.Team_2]["points"] += 1
        result = {fixture.Team_1 : goals_1, fixture.Team_2 : goals_2}
        return result

    def play_all_matches(self, fixture_df):
        """
        Given the full DataFrame full of fixtures, find the ones that correspond
        to this group, and use them to fill our list of results

        Parameters
        ==========
        fixture_df: pandas DataFrame of all fixtures, with Team_1 and Team_2 columns
        """
        for _, fixture in fixture_df.iterrows():
            if fixture.Team_1 in self.teams and fixture.Team_2 in self.teams:
                print(f"{fixture.Team_1} vs {fixture.Team_2}")
                result = self.play_match(fixture)
                self.results.append(result)


    def get_qualifiers(self):
        """
        return the two teams that topped the group
        """
        if len(self.results) < 6:
            print(f"Group {self.name} not finished yet - only {len(self.results)} matches played")
            return
        sorted_table = self.sort_table()
        return sorted_table[0]["team"], sorted_table[1]["team"]


    def sort_two_teams(self, team_1, team_2):
        """
        Decide which of two teams is ahead in the table, according to rules:
        1) most points
        2) best goal difference
        3) most goals scored
        4) head-to-head
        5) random

        Returns
        =======
        team_1, team_2: str, in order of higher-to-lower placing
        """
        # decide by points
        if self.table[team_1]["points"] > self.table[team_2]["points"]:
            return team_1, team_2
        elif self.table[team_1]["points"] < self.table[team_2]["points"]:
            return team_2, team_1
        # decide by goal difference
        elif (self.table[team_1]["goals_for"] - self.table[team_1]["goals_for"])   >\
              (self.table[team_2]["goals_for"] - self.table[team_2]["goals_for"]):
               return team_1, team_2
        elif (self.table[team_1]["goals_for"] - self.table[team_1]["goals_for"])   <\
             (self.table[team_2]["goals_for"] - self.table[team_2]["goals_for"]):
              return team_2, team_1
        # decide by goals scored
        elif self.table[team_1]["goals_for"] > self.table[team_2]["goals_for"]:
              return team_1, team_2
        elif self.table[team_1]["goals_for"] < self.table[team_2]["goals_for"]:
              return team_2, team_1
        # decide by head-to-head
        else:
            for result in self.results:
                if list(result.keys()) == [team_1, team_2]:
                    if result[team_1] > result[team_2]:
                        return team_1, team_2
                    elif result[team_2] > result[team_1]:
                        return team_2, team_1
                    break
        # decide by random
        if random.random() > 0.5:
            return team_2, team_1
        else:
            return team_1, team_2

    def sort_table(self):
        team_list = [{"team": k, **v} for k,v in self.table.items()]
        team_list = sorted(team_list, key=lambda t: t['points'],reverse=True)
        # easy situation first
        if team_list[0]["points"] > team_list[1]["points"] and team_list[1]["points"] > team_list[2]["points"]:
            return team_list
        else:
            # see if we need to swap 2nd and 3rd
            if team_list[1]["points"] == team_list[2]["points"]:
                new_team_1, new_team_2 = self.sort_two_teams(team_list[1]["team"], team_list[2]["team"])
                if new_team_1 != team_list[1]["team"]:
                    tmp_team = team_list[1]
                    team_list[1] = team_list[2]
                    team_list[2] = tmp_team
            # see if we need to swap 1st and 2nd
            if team_list[0]["points"] == team_list[1]["points"]:
                new_team_0, new_team_1 = self.sort_two_teams(team_list[0]["team"], team_list[1]["team"])
                if new_team_0 != team_list[0]["team"]:
                    tmp_team = team_list[0]
                    team_list[0] = team_list[1]
                    team_list[1] = tmp_team
        return team_list


    def __str__(self):
        max_team_name_length = 0
        for t in self.teams:
            if len(t) > max_team_name_length:
                max_team_name_length = len(t)

        output = f"  Team{' '*(max_team_name_length-8)}| Points | GS |  GA \n"
        team_list = self.sort_table()
        for t in team_list:
            output += f" {t['team']}{' '*(max_team_name_length-len(t['team']))} {t['points']}      {t['goals_for']}     {t['goals_against']} \n"
        return output


class Tournament:
    def __init__(self):
        self.teams_df = pd.read_csv("../data/teams.csv")
        self.fixtures_df = pd.read_csv("../data/fixtures.csv")
        self.group_names = list(set(self.teams_df["Group"].values))
        self.groups = []
        for n in self.group_names:
            g = Group(n, list(self.teams_df[self.teams_df["Group"]==n].Team.values))
            self.groups.append(g)
        self.aliases = {}


    def add_result(team_1, team_2, score_1, score_2, stage):
        """
        Enter a match result explicitly

        Parameters
        ==========
        team_1, team_2: both str, names of teams, as in teams.csv
        score_1, score_2: both int, scores of respective teams
        stage: str, must be "Group", "R16", "QF", "SF", "F"
        """
        # find the fixture
        for idx, row in self.fixtures_df:
            if stage != row.Stage:
                continue


    def play_group_stage(self):
        for g in self.groups:
            g.play_all_matches(self.fixtures_df)


    def play_knockout_stages(self):
        """
        For the round of 16, assign the first and second place teams
        from each group to the aliases e.g. "A1", "B2"
        """
        for g in self.groups:
            if len(g.results) != 6:
                print(f" Group {g.name} has only played {len(g.results)} matches")
            t1, t2 = g.get_qualifiers()
            self.aliases["1"+g.name] = t1
            self.aliases["2"+g.name] = t2
        for stage in ["R16","QF","SF","F"]:
            for _, f in self.fixtures_df.iterrows():
                if f.Stage == stage:
                    self.aliases[f.Team_1+f.Team_2] = predict_knockout_match(self.aliases[f.Team_1], self.aliases[f.Team_2])
        for k,v in self.aliases.items():
            if len(k) == 32:
                self.winner = v
        print(f"====== WINNER: {self.winner} =======")
