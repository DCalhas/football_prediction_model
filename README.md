# football_prediction_model
Model that is able to predict the result between two teams

The objective of this project is to construct different models, that are trained for different competitions,
being the first competition Primeira Liga Portuguesa.

Good sites for football statistics:
1. zeroazero
2. http://www.football-data.co.uk/portugalm.php
3. https://www.statbunker.com/
4. http://publications.lib.chalmers.se/records/fulltext/250411/250411.pdf
5. http://pytorch.org/

__Maybe using the data from the game Football Manager is not a bad idea__

For presentation: https://jupyter.org/

The how it is going to be able to predict is by the following way:
There is going to be data provided in csv format, from previous years. The optimal amount should be more than 10 seasons of stats.
Stats to be taken into account:
* Transfer money spent in the current year (home/away)
* The market value of the team (this can be found in the transfer market site) (home/away)
* The league rank, current position prior to the game (home/away)
* The number of the fixture, in Primeira Liga Portuguesa there are 34 fixtures
* Away goals scored (home/away)
* Away goals conceded (home/away)
* Home goals scored (home/away)
* Home goals conceded (home/away)
* Total wins in the last 10 games (home/away)
* Total draws in the last 10 games (home/away)
* Total losses in the last 10 games (home/away)
* League points (home/away) should be normalized
* Home ground first team (shall be a boolean 0 or 1)
* Home ground second team (shall be a boolean 0 or 1)
* Clean sheets (home/away) should be normalized
