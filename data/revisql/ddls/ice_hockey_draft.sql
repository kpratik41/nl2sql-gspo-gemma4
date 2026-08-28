CREATE TABLE weight_info (
    `weight_id` integer, -- weight id. the unique weight id.
    `weight_in_kg` integer, -- weight in kg. e.g. 70: -->70 kg.
    `weight_in_lbs` integer, -- weight in lbs.
    primary key (weight_id)
);

CREATE TABLE height_info (
    `height_id` integer, -- height id. the unique height id.
    `height_in_cm` integer, -- height in cm. e.g. 180 --> the height is 180 cm.
    `height_in_inch` text, -- height in inch.
    primary key (height_id)
);

CREATE TABLE SeasonStatus (
    `ELITEID` integer, -- ELITE ID. the id number of the players.
    `SEASON` text, -- season when players are playing.
    `TEAM` text, -- which team the player belong to.
    `LEAGUE` text, -- league.
    `GAMETYPE` text, -- GAME TYPE. type of games.  Regular season  playoffs (post season).
    `GP` integer, -- Game Plays. number of games.
    `G` integer, -- Goals in Draft Year.
    `A` integer, -- Assists in Draft Year.
    `P` integer, -- Points in Draft Year. higher --> more valuable.
    `PIM` integer, -- Penalty Minutes. Penalty Minutes in Draft Year. higher --> This player has committed more rule violations.
    `PLUSMINUS` integer, -- Plus Minutes. Goal Differential in Draft Year. Goal Differential.
    foreign key (ELITEID) references PlayerInfo(ELITEID)
);

CREATE TABLE PlayerInfo (
    `ELITEID` integer, -- ELITE ID. the unique number identifying the players who attended the draft.
    `PlayerName` text, -- Player Name. the name of the player.
    `birthdate` text, -- the birthdate of the player.
    `birthyear` date, -- the birth year of the player.
    `birthmonth` integer, -- the birth month of the player.
    `birthday` integer, -- the birthday of the player.
    `birthplace` text, -- the player of the birthplace.
    `nation` text, -- the nation of the player. can ask questions about their corresponding continents. or group nations with their continents. You can refer to https://worldpopulationreview.com/country-rankings/list-of-countries-by-continent. e.g.: Slovakia --> Europe.
    `height` integer, -- the id number identifying heights.
    `weight` integer, -- the id number identifying weights.
    `position_info` text, -- position information. position information of the player. There are six different positions in hockey:. left wing,. right wing,. center,. left defenseman,. right defenseman. goalie. Left wings, right wings, and centers are all considered forwards, while left and right defensemen are considered the defense.
    `shoots` text, --  L: Left-shooted.  R: Right-shooted.  '-': no preference.
    `draftyear` integer, -- draft year.
    `draftround` integer, -- draft round.
    `overall` integer, -- overall orders of draft picks.
    `overallby` text, -- drafted by which team.
    `CSS_rank` integer, -- Central Scouting Service ranking. higher rank refers to higher prospects for the draft.
    `sum_7yr_GP` integer, -- sum 7-year game plays. Total NHL games played in players first 7 years of NHL career. higher --> more attendance in the first 7 years.
    `sum_7yr_TOI` integer, -- sum 7-year time on ice. Total NHL Time on Ice in players first 7 years of NHL career. higher --> more playing time in the first 7 years of career.
    `GP_greater_than_0` text, -- game play greater than 0. Played a game or not in players first 7 years of NHL career.  yes.  no.
    primary key (ELITEID),
    foreign key (height) references height_info(height_id),
    foreign key (weight) references weight_info(weight_id)
);

