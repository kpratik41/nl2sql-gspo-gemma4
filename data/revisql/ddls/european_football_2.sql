CREATE TABLE Team_Attributes (
    `id` integer, -- the unique id for teams.
    `team_fifa_api_id` integer, -- team federation international football association api id. the id of the team fifa api.
    `team_api_id` integer, -- team api id. the id of the team api.
    `date` text, -- Date. e.g. 2010-02-22 00:00:00.
    `buildUpPlaySpeed` integer, -- build Up Play Speed. the speed in which attacks are put together. the score which is between 1-00 to measure the team's attack speed.
    `buildUpPlaySpeedClass` text, -- build Up Play Speed Class. the speed class. commonsense reasoning:. • Slow: 1-33. • Balanced: 34-66. • Fast: 66-100.
    `buildUpPlayDribbling` integer, -- build Up Play Dribbling. the tendency/ frequency of dribbling.
    `buildUpPlayDribblingClass` text, -- build Up Play Dribbling Class. the dribbling class. commonsense reasoning:. • Little: 1-33. • Normal: 34-66. • Lots: 66-100.
    `buildUpPlayPassing` integer, -- build Up Play Passing. affects passing distance and support from teammates.
    `buildUpPlayPassingClass` text, -- build Up Play Passing Class. the passing class. commonsense reasoning:. • Short: 1-33. • Mixed: 34-66. • Long: 66-100.
    `buildUpPlayPositioningClass` text, -- build Up Play Positioning Class. A team's freedom of movement in the 1st two thirds of the pitch. Organised / Free Form.
    `chanceCreationPassing` integer, -- chance Creation Passing. Amount of risk in pass decision and run support.
    `chanceCreationPassingClass` text, -- chance Creation Passing Class. the chance creation passing class. commonsense reasoning:. • Safe: 1-33. • Normal: 34-66. • Risky: 66-100.
    `chanceCreationCrossing` integer, -- chance Creation Crossing. The tendency / frequency of crosses into the box.
    `chanceCreationCrossingClass` text, -- chance Creation Crossing Class. the chance creation crossing class. commonsense reasoning:. • Little: 1-33. • Normal: 34-66. • Lots: 66-100.
    `chanceCreationShooting` integer, -- chance Creation Shooting. The tendency / frequency of shots taken.
    `chanceCreationShootingClass` text, -- chance Creation Shooting Class. the chance creation shooting class. commonsense reasoning:. • Little: 1-33. • Normal: 34-66. • Lots: 66-100.
    `chanceCreationPositioningClass` text, -- chance Creation Positioning Class. A team’s freedom of movement in the final third of the pitch. Organised / Free Form.
    `defencePressure` integer, -- defence Pressure. Affects how high up the pitch the team will start pressuring.
    `defencePressureClass` text, -- defence Pressure Class. the defence pressure class. commonsense reasoning:. • Deep: 1-33. • Medium: 34-66. • High: 66-100.
    `defenceAggression` integer, -- defence Aggression. Affect the team’s approach to tackling the ball possessor.
    `defenceAggressionClass` text, -- defence Aggression Class. the defence aggression class. commonsense reasoning:. • Contain: 1-33. • Press: 34-66. • Double: 66-100.
    `defenceTeamWidth` integer, -- defence Team Width. Affects how much the team will shift to the ball side.
    `defenceTeamWidthClass` text, -- defence Team Width Class. the defence team width class. commonsense reasoning:. • Narrow: 1-33. • Normal: 34-66. • Wide: 66-100.
    `defenceDefenderLineClass` text, -- defence Defender Line Class. Affects the shape and strategy of the defence. Cover/ Offside Trap.
    primary key (id)
);

CREATE TABLE Player (
    `id` integer, -- the unique id for players.
    `player_api_id` integer, -- player api id. the id of the player api.
    `player_name` text, -- player name.
    `player_fifa_api_id` integer, -- player federation international football association api id. the id of the player fifa api.
    `birthday` text, -- the player's birthday. e.g. 1992-02-29 00:00:00. commonsense reasoning:. Player A is older than player B means that A's birthday is earlier than B's.
    `height` integer, -- the player's height.
    `weight` integer, -- the player's weight.
    primary key (id)
);

CREATE TABLE Match (
    `id` integer, -- the unique id for matches.
    `country_id` integer, -- country id.
    `league_id` integer, -- league id.
    `season` text, -- the season of the match.
    `stage` integer, -- the stage of the match.
    `date` text, -- the date of the match. e.g. 2008-08-17 00:00:00.
    `match_api_id` integer, -- match api id. the id of the match api.
    `home_team_api_id` integer, -- home team api id. the id of the home team api.
    `away_team_api_id` integer, -- away team api id. the id of the away team api.
    `home_team_goal` integer, -- home team goal. the goal of the home team.
    `away_team_goal` integer, -- away team goal. the goal of the away team.
    `home_player_X1` integer, -- No description.
    `home_player_X2` integer, -- No description.
    `home_player_X3` integer, -- No description.
    `home_player_X4` integer, -- No description.
    `home_player_X5` integer, -- No description.
    `home_player_X6` integer, -- No description.
    `home_player_X7` integer, -- No description.
    `home_player_X8` integer, -- No description.
    `home_player_X9` integer, -- No description.
    `home_player_X10` integer, -- No description.
    `home_player_X11` integer, -- No description.
    `away_player_X1` integer, -- No description.
    `away_player_X2` integer, -- No description.
    `away_player_X3` integer, -- No description.
    `away_player_X4` integer, -- No description.
    `away_player_X5` integer, -- No description.
    `away_player_X6` integer, -- No description.
    `away_player_X7` integer, -- No description.
    `away_player_X8` integer, -- No description.
    `away_player_X9` integer, -- No description.
    `away_player_X10` integer, -- No description.
    `away_player_X11` integer, -- No description.
    `home_player_Y1` integer, -- No description.
    `home_player_Y2` integer, -- No description.
    `home_player_Y3` integer, -- No description.
    `home_player_Y4` integer, -- No description.
    `home_player_Y5` integer, -- No description.
    `home_player_Y6` integer, -- No description.
    `home_player_Y7` integer, -- No description.
    `home_player_Y8` integer, -- No description.
    `home_player_Y9` integer, -- No description.
    `home_player_Y10` integer, -- No description.
    `home_player_Y11` integer, -- No description.
    `away_player_Y1` integer, -- No description.
    `away_player_Y2` integer, -- No description.
    `away_player_Y3` integer, -- No description.
    `away_player_Y4` integer, -- No description.
    `away_player_Y5` integer, -- No description.
    `away_player_Y6` integer, -- No description.
    `away_player_Y7` integer, -- No description.
    `away_player_Y8` integer, -- No description.
    `away_player_Y9` integer, -- No description.
    `away_player_Y10` integer, -- No description.
    `away_player_Y11` integer, -- No description.
    `home_player_1` integer, -- No description.
    `home_player_2` integer, -- No description.
    `home_player_3` integer, -- No description.
    `home_player_4` integer, -- No description.
    `home_player_5` integer, -- No description.
    `home_player_6` integer, -- No description.
    `home_player_7` integer, -- No description.
    `home_player_8` integer, -- No description.
    `home_player_9` integer, -- No description.
    `home_player_10` integer, -- No description.
    `home_player_11` integer, -- No description.
    `away_player_1` integer, -- No description.
    `away_player_2` integer, -- No description.
    `away_player_3` integer, -- No description.
    `away_player_4` integer, -- No description.
    `away_player_5` integer, -- No description.
    `away_player_6` integer, -- No description.
    `away_player_7` integer, -- No description.
    `away_player_8` integer, -- No description.
    `away_player_9` integer, -- No description.
    `away_player_10` integer, -- No description.
    `away_player_11` integer, -- No description.
    `goal` text, -- the goal of the match.
    `shoton` text, -- shot on. the shot on goal of the match. commonsense reasoning:. A shot on goal is a shot that enters the goal or would have entered the goal if it had not been blocked by the goalkeeper or another defensive player.
    `shotoff` text, -- shot off. the shot off goal of the match, which is the opposite of shot on.
    `foulcommit` text, -- foul commit. the fouls occurred in the match.
    `card` text, -- the cards given in the match.
    `cross` text, -- Balls sent into the opposition team's area from a wide position in the match.
    `corner` text, -- Ball goes out of play for a corner kick in the match.
    `possession` text, -- The duration from a player taking over the ball in the match.
    `B365H` real, -- No description.
    `B365D` real, -- No description.
    `B365A` real, -- No description.
    `BWH` real, -- No description.
    `BWD` real, -- No description.
    `BWA` real, -- No description.
    `IWH` real, -- No description.
    `IWD` real, -- No description.
    `IWA` real, -- No description.
    `LBH` real, -- No description.
    `LBD` real, -- No description.
    `LBA` real, -- No description.
    `PSH` real, -- No description.
    `PSD` real, -- No description.
    `PSA` real, -- No description.
    `WHH` real, -- No description.
    `WHD` real, -- No description.
    `WHA` real, -- No description.
    `SJH` real, -- No description.
    `SJD` real, -- No description.
    `SJA` real, -- No description.
    `VCH` real, -- No description.
    `VCD` real, -- No description.
    `VCA` real, -- No description.
    `GBH` real, -- No description.
    `GBD` real, -- No description.
    `GBA` real, -- No description.
    `BSH` real, -- No description.
    `BSD` real, -- No description.
    `BSA` real, -- No description.
    primary key (id)
);

CREATE TABLE League (
    `id` integer, -- the unique id for leagues.
    `country_id` integer, -- country id. the unique id for countries.
    `name` text, -- league name.
    primary key (id)
);

CREATE TABLE Country (
    `id` integer, -- the unique id for countries.
    `name` text, -- country name.
    primary key (id)
);

CREATE TABLE Player_Attributes (
    `id` integer, -- the unique id for players.
    `player_fifa_api_id` integer, -- player federation international football association api id. the id of the player fifa api.
    `player_api_id` integer, -- player api id. the id of the player api.
    `date` text, -- date. e.g. 2016-02-18 00:00:00.
    `overall_rating` integer, -- the overall rating of the player. commonsense reasoning:. The rating is between 0-100 which is calculated by FIFA. Higher overall rating means the player has a stronger overall strength.
    `potential` integer, -- potential of the player. commonsense reasoning:. The potential score is between 0-100 which is calculated by FIFA. Higher potential score means that the player has more potential.
    `preferred_foot` text, -- preferred foot. the player's preferred foot when attacking. right/ left.
    `attacking_work_rate` text, -- attacking work rate. the player's attacking work rate. commonsense reasoning:. • high: implies that the player is going to be in all of your attack moves. • medium: implies that the player will select the attack actions he will join in. • low: remain in his position while the team attacks.
    `defensive_work_rate` text, -- the player's defensive work rate. commonsense reasoning:. • high: remain in his position and defense while the team attacks. • medium: implies that the player will select the defensive actions he will join in. • low: implies that the player is going to be in all of your attack moves instead of defensing.
    `crossing` integer, -- the player's crossing score. commonsense reasoning:. Cross is a long pass into the opponent's goal towards the header of sixth-yard teammate. The crossing score is between 0-100 which measures the tendency/frequency of crosses in the box. Higher potential score means that the player performs better in crossing actions.
    `finishing` integer, -- the player's finishing rate. 0-100 which is calculated by FIFA.
    `heading_accuracy` integer, -- heading accuracy. the player's heading accuracy. 0-100 which is calculated by FIFA.
    `short_passing` integer, -- short passing. the player's short passing score. 0-100 which is calculated by FIFA.
    `volleys` integer, -- the player's volley score. 0-100 which is calculated by FIFA.
    `dribbling` integer, -- the player's dribbling score. 0-100 which is calculated by FIFA.
    `curve` integer, -- the player's curve score. 0-100 which is calculated by FIFA.
    `free_kick_accuracy` integer, -- free kick accuracy. the player's free kick accuracy. 0-100 which is calculated by FIFA.
    `long_passing` integer, -- long passing. the player's long passing score. 0-100 which is calculated by FIFA.
    `ball_control` integer, -- ball control. the player's ball control score. 0-100 which is calculated by FIFA.
    `acceleration` integer, -- the player's acceleration score. 0-100 which is calculated by FIFA.
    `sprint_speed` integer, -- sprint speed. the player's sprint speed. 0-100 which is calculated by FIFA.
    `agility` integer, -- the player's agility. 0-100 which is calculated by FIFA.
    `reactions` integer, -- the player's reactions score. 0-100 which is calculated by FIFA.
    `balance` integer, -- the player's balance score. 0-100 which is calculated by FIFA.
    `shot_power` integer, -- shot power. the player's shot power. 0-100 which is calculated by FIFA.
    `jumping` integer, -- the player's jumping score. 0-100 which is calculated by FIFA.
    `stamina` integer, -- the player's stamina score. 0-100 which is calculated by FIFA.
    `strength` integer, -- the player's strength score. 0-100 which is calculated by FIFA.
    `long_shots` integer, -- long shots. the player's long shots score. 0-100 which is calculated by FIFA.
    `aggression` integer, -- the player's aggression score. 0-100 which is calculated by FIFA.
    `interceptions` integer, -- the player's interceptions score. 0-100 which is calculated by FIFA.
    `positioning` integer, -- the player's. positioning score. 0-100 which is calculated by FIFA.
    `vision` integer, -- the player's vision score. 0-100 which is calculated by FIFA.
    `penalties` integer, -- the player's penalties score. 0-100 which is calculated by FIFA.
    `marking` integer, -- the player's markingscore. 0-100 which is calculated by FIFA.
    `standing_tackle` integer, -- standing tackle. the player's standing tackle score. 0-100 which is calculated by FIFA.
    `sliding_tackle` integer, -- sliding tackle. the player's sliding tackle score. 0-100 which is calculated by FIFA.
    `gk_diving` integer, -- goalkeep diving. the player's goalkeep diving score. 0-100 which is calculated by FIFA.
    `gk_handling` integer, -- goalkeep handling. the player's goalkeep diving score. 0-100 which is calculated by FIFA.
    `gk_kicking` integer, -- goalkeep kicking. the player's goalkeep kicking score. 0-100 which is calculated by FIFA.
    `gk_positioning` integer, -- goalkeep positioning. the player's goalkeep positioning score. 0-100 which is calculated by FIFA.
    `gk_reflexes` integer, -- goalkeep reflexes. the player's goalkeep reflexes score. 0-100 which is calculated by FIFA.
    primary key (id)
);

CREATE TABLE Team (
    `id` integer, -- the unique id for teams.
    `team_api_id` integer, -- team api id. the id of the team api.
    `team_fifa_api_id` integer, -- team federation international football association api id. the id of the team fifa api.
    `team_long_name` text, -- team long name. the team's long name.
    `team_short_name` text, -- team short name. the team's short name.
    primary key (id)
);

