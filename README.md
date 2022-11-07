# WorldCupPrediction
Predicting results for the 2022 world cup.

Matches are predicted using a framework based on the team-level model in https://github.com/alan-turing-institute/AIrsenal, which in turn uses https://github.com/anguswilliams91/bpl-next.
This model is trained on international mens football results obtained from https://github.com/martj42/international_results.
The original model is a version of [Dixon and Coles](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9876.00065).

## Installation

The easiest way to use the code is via [poetry](https://python-poetry.org/).  If you have poetry installed, from this directory, you can do
```
poetry shell
poetry install
```
to first open a shell in a virtual environment, and then install the dependencies and the `wcpredictor` package.

## Usage

### Simulating a tournament multiple times

There are a couple of command-line applications that can be run when the `wcpredictor` package is installed as described above.

In order to simulate the tournament N times, you can do
```
wcpred_run_simulations --num_simulations <N> --tournament_year <year> --training_data_start <YYYY-MM-DD> --training_data_end <YYYY-MM-DD> --output_csv <outputfilename> --use_ratings
```
and the results, in the form of a table of how many times each team got to each stage of the competition, will be saved in the specified csv file.   At present, the allowed values for `tournament_year` are "2014", "2018", and "2022" (the default).

Once you have a csv file saved from running that, you can plot the top ten most frequent winners by running:
```
wcpred_plot_winners --input_csv <inputfilename> --output_png <outputfilename>
```
and the results will be saved in the specified png.

You can also make a plot showing how far in the tournament a selection of teams got, by running e.g.:
```
wcpred_plot_progress --input_csv <inputfilename> --output_png <outputfilename> --team_list "England,Wales"
```

Note that both these commands can be run with `--help` to see the options.

### Running a single tournament

In a python session, you can do something like:
```
python
>>> from wcpredictor import Tournament, get_and_train_model
>>> t = Tournament("2022") # can also choose "2018" or "2014"
>>> model = get_and_train_model(start_date="2016-06-01", end_date="2022-11-20") # choose dates for training data
>>> t.play_group_stage(model)
>>> # at this stage, we can look at how each group is doing
>>> print(t.groups["A"])
Position |  Team   | Points | GS |  GA
   1st    Netherlands   6      8     4
   2nd    Qatar         4      2     1
   3rd    Ecuador       4      1     5
   4th    Senegal       3      4     5
>>> # or, we can go ahead and play the knockout stages
>>> t.play_knockout_stages(model)
>>> print(t.winner)
```
