# WorldCupPrediction
Predicting results for the 2022 world cup.  

Matches are predicted using a framework based on the team-level model in [https://github.com/alan-turing-institute/AIrsenal], which in turn uses [https://github.com/anguswilliams91/bpl-next].
This model is trained on international mens football results from the past four years.  This data, along with lists of teams and fixtures for the 2022 world cup, can be found in the `data/` directory.

The structure of the tournament can be found in `src/tournament.py`.  This contains the logic for determining which teams qualify from groups, and who faces who in the knockout stages.

