CREATE TABLE series_post (
    `id` integer, -- unique number of this record.
    `year` integer, -- year.
    `round` text, -- round. known abbreviations：. F：final. SF: semi-final. QF：quater-final 1/4. DF：division-final. DSF：division semi-final.
    `series` text, -- not useful.
    `tmIDWinner` text, -- team id winner.
    `lgIDWinner` text, -- league id winner.
    `tmIDLoser` text, -- team id loser.
    `lgIDLoser` text, -- league id loser.
    `W` integer, -- wins.
    `L` integer, -- loses.
    primary key (id),
    foreign key (tmIDWinner, year) references teams (tmID, year)
    foreign key (tmIDLoser, year) references teams (tmID, year)
);

CREATE TABLE teams (
    `year` integer, -- year.
    `lgID` text, -- league ID. league name. main categories:. NBA, ABA.
    `tmID` text, -- team ID. team id. abbreviated name.
    `franchID` text, -- not useful.
    `confID` text, -- not useful.
    `divID` text, -- division ID.
    `rank` integer, -- rank. less is better.
    `confRank` integer, -- not useful.
    `playoff` text, -- another name: post season. for brevity, if the value is not null or empty, it means this team attends the playoffs, otherwise, means not attend.
    `name` text, -- full name of the team.
    `o_fgm` integer, -- offense field goal made. how many offense field goal made by this team in this season.
    `o_ftm` integer, -- offense free throw made. how many offense free throw made by this team in this season.
    `o_pts` integer, -- offense points.
    `d_pts` integer, -- defense points.
    `homeWon` integer, -- home wins. wins in the home.
    `homeLost` integer, -- home loses. loses in the home.
    `awayWon` integer, -- away wins. wins in the away.
    `awayLost` integer, -- away loses. loses in the away.
    `won` integer, -- total wins of this team in this year.
    `lost` integer, -- total losts of this team in this year.
    `games` integer, -- total number of games that this team played in this year (season).
    `arena` text, -- arena. null or empty refers to the fact that this team doesn't have individual arenas, otherwise, it has individual arenas.
    primary key (year, tmID)
);

CREATE TABLE players (
    `playerID` text, -- ID number identifying the unique number of the player.
    `useFirst` text, -- use first name. not useful.
    `firstName` text, -- the first name of the player.
    `middleName` text, -- the middle name of the player.
    `lastName` text, -- the last name of the player.
    `nameGiven` text, -- not useful.
    `fullGivenName` text, -- not useful.
    `nameSuffix` text, -- the suffix name of the player.
    `nameNick` text, -- nick name.
    `pos` text, -- position. the position of the player. C: Center. F: Forward. G: Guard. some player can have two positions simultaneously.
    `firstseason` integer, -- nothing special.
    `lastseason` integer, -- nothing special.
    `height` real, -- inch.
    `weight` integer, -- lb. null or empty means it doesn't have this information.
    `college` text, -- college. null or empty means it doesn't have this information. which colledge does this player graduate from.
    `collegeOther` text, -- the college that transferred from. some players may have the record of transferring from other colleges.
    `birthDate` date, -- birthdate.
    `birthCity` text, -- birthcity. null / empty refers to not recorded.
    `birthState` text, -- birth state.
    `birthCountry` text, -- birth country.
    `highSchool` text, -- high school. null / empty refers to not recorded.
    `hsCity` text, -- high school city. null / empty refers to not recorded.
    `hsState` text, -- high school state. null / empty refers to not recorded.
    `hsCountry` text, -- high school country. null / empty refers to not recorded.
    `deathDate` date, -- deathdate. 0000-00-00 means this player is still alive. within clear date：. 2011-11-10, which means this guy was died on "2011-11-10".
    `race` text, -- B：black,. W: white. null or empty: nothing.
    primary key (playerID)
);

CREATE TABLE awards_players (
    `playerID` text, -- ID number identifying the unique number of the player.
    `award` text, -- the name of the award.
    `year` integer, -- the time of this award.
    `lgID` text, -- league ID. the name of the league. mainly categories: "NBA" & "ABA".
    `note` text, -- notification. null refers to that no special things.
    `pos` text, -- position. the position of the player. C: Center. F: Forward. G: Guard.
    primary key (playerID, year, award),
    foreign key (playerID) references players (playerID)
);

CREATE TABLE draft (
    `id` integer, -- the unique number to determine one row of the data.
    `draftYear` integer, -- which year does this draft happens.
    `draftRound` integer, -- the Round number of this draft. If the value is 0, it means that there is no draft in this year. 1 refers that this player was picked in the first round.
    `draftSelection` integer, -- league ID. the position that the player was picked. 0：This signifies that it doesn't contain the draft information of this player. 7：this player was selected in the 7th position in this round. draftRound: 1; draftSelection: 7. represents: this player was selected in the round 1, 7th position.
    `draftOverall` integer, -- draft overall rank. The player's overall draft position for the year. this is the overall draft rank of the player.
    `tmID` text, -- team ID. team name. abbreviated name.
    `firstName` text, -- the first name of the player.
    `lastName` text, -- the last name of the player.
    `suffixName` text, -- the suffix name of the player.
    `playerID` text, -- ID number identifying the unique number of the player.
    `draftFrom` text, -- the university that the drafted players come from. the name of the university.
    `lgID` text, -- league ID. the league name. mainly categories: "NBA" & "ABA".
    primary key (id),
    foreign key (tmID, draftYear) references teams (tmID, year)
);

CREATE TABLE players_teams (
    `id` integer, -- the unique number to identify this record.
    `playerID` text, -- ID number identifying the unique number of the player.
    `year` integer, -- year.
    `stint` integer, -- the period of the time player spent played for this team.
    `tmID` text, -- team ID. team name. abbreviated name.
    `lgID` text, -- No description.
    `GP` integer, -- game presentatons. game presentatons (attendance). min: 0, max: 82. if this number is equal to 82, means full attendance.
    `GS` integeer, -- game starting. min: 0, max: 82,. when GP = GS, it means that this player plays as the stating lineup fully.
    `minutes` integer, -- minutes.
    `points` integer, -- points.
    `oRebounds` integer, -- offense rebounds. empty or null refers to none offense rebounds，. 1: one offense rebound.
    `dRebounds` integer, -- defense rebounds. empty or null refers to none defense rebounds，. 1: one defense rebound.
    `rebounds` integer, -- total rebounds. empty or null refers to none total rebounds，. 3: totally gets 3 rebounds including offense and defense rebounds. commensense evidence:. total rebounds = offense rebounds + defense rebounds.
    `assists` integer, -- assistants. null or empty refers to none. 2: 2 assistants.
    `steals` integer, -- steals. null or empty refers to none. 2: 2 steals.
    `blocks` integer, -- blocks. null or empty refers to none. 2: 2 blocks.
    `turnovers` integer, -- turnovers. null or empty refers to none. 2: 2 turnovers.
    `PF` integer, -- personal fouls. null or empty refers to none. 2: 2 personal fouls.
    `fgAttempted` integer, -- field goal attempted. null or empty refers to none. 2: 2 field goal attempts.
    `fgMade` integer, -- field goal made. null or empty refers to none. 2: 2 field goal made.
    `ftAttempted` integer, -- free throw attempted. null or empty refers to none. 2: 2 free throw attempts.
    `ftMade` integer, -- free throw made. null or empty refers to none. 2: 2 free throw made.
    `threeAttempted` integer, -- three point attempted. null or empty refers to none. 2: 2 three point attempts.
    `threeMade` integer, -- three point made. null or empty refers to none. 2: 2 three point made.
    `PostGP` integer, -- post season game presentations. 0: this player doesn't present in the post season (playoffs).
    `PostGS` integer, -- post season game starting.
    `PostMinutes` integer, -- post season minutes.
    `PostPoints` integer, -- post season points.
    `PostoRebounds` integer, -- post season offense rebounds. null or empty refers to none. 1: 1 offense rebounds in the post season.
    `PostdRebounds` integer, -- post season defense rebounds. null or empty refers to none. 1: 1 defense rebounds in the post season.
    `PostRebounds` integer, -- post season defense rebounds. null or empty refers to none. 3: 3 rebounds in the post season totally.
    `PostAssists` integer, -- post season assistants. null or empty refers to none. 1: 1 assistance in the post season.
    `PostSteals` integer, -- post season steals. null or empty refers to none. 1: 1 offense steals in the post season.
    `PostBlocks` integer, -- post season blocks. null or empty refers to none. 1: 1 block in the post season.
    `PostTurnovers` integer, -- post season turnovers. null or empty refers to none. 1: 1 turnover in the post season.
    `PostPF` integer, -- post season personal fouls. null or empty refers to none. 1: 2 personal fouls in the post season.
    `PostfgAttempted` integer, -- post season field goal attempted. null or empty refers to none. 1: 1 field goal attempts in the post season.
    `PostfgMade` integer, -- post season field goal made. null or empty refers to none. 1: 1 field goal made in the post season.
    `PostftAttempted` integer, -- post season field free throw attempted. null or empty refers to none. 1: 1 free throw attempts in the post season.
    `PostftMade` integer, -- post season free throw made. null or empty refers to none. 1: 1 free throw made in the post season.
    `PostthreeAttempted` integer, -- post season three point attempted. null or empty refers to none. 1: 1 three point attempts in the post season.
    `PostthreeMade` integer, -- post season three point made. null or empty refers to none. 1: 1 three point made in the post season.
    `note` text, -- No description.
    primary key (id),
    foreign key (tmID, year) references teams (tmID, year)
);

CREATE TABLE coaches (
    `coachID` text, -- ID number identifying the unique number of the coach.
    `year` integer, -- No description.
    `tmID` text, -- team ID. team name. abbreviated name.
    `lgID` text, -- league ID. league name. mainly categories: "NBA" & "ABA".
    `stint` integer, -- the period of the time coaching this team.
    `won` integer, -- the number of won games. 0: no wins.
    `lost` integer, -- the number of lost games. 0: win all the games.
    `post_wins` integer, -- post season wins. the number of games won in the post-season (playoffs) games. 0: no wins. If the team's post-season wins and losses are all zero, it implies that the team did not participate in the post-season (playoffs) that year.
    `post_losses` integer, -- post season losses. the number of games lost in the post-season (playoffs) games. 0: win all the games. If the team's post-season wins and losses are all zero, it implies that the team did not participate in the post-season (playoffs) that year.
    primary key (coachID, year, tmID, stint),
    foreign key (tmID, year) references teams (tmID, year)
);

CREATE TABLE awards_coaches (
    `id` integer, -- id of this row of data.
    `year` integer, -- which year did the coach receive the honor?
    `coachID` text, -- id number of the coaches.
    `award` text, -- the award names. mainly categories: "NBA" & "ABA".
    `lgID` text, -- league ID. the name of the league. mainly categories: "NBA" & "ABA".
    `note` text, -- special notification. null" refers to that no special things.
    primary key (id),
    foreign key (coachID, year) references coaches (coachID, year)
);

CREATE TABLE player_allstar (
    `playerID` text, -- ID number identifying the unique number of the player.
    `first_name` text, -- the first name of the player.
    `last_name` text, -- the last name of the player.
    `season_id` integer, -- the id number of the season.
    `conference` text, -- which conference that players belong to. two categories: west; east.
    `league_id` text, -- league ID. the league name. two categories: NBA; ABA.
    `games_played` integer, -- how many all star games that this player played in this season. mostly it's 1.
    `minutes` integer, -- minutes of attendance. 18：played in 18 minutes.
    `points` integer, -- points. 19: get 19 points；. null --> doesn't get points.
    `o_rebounds` integer, -- offense rebounds. empty or null refers to none offense rebounds，. 1: one offense rebound.
    `d_rebounds` integer, -- defense rebounds. empty or null refers to none defense rebounds，. 1: one defense rebound.
    `rebounds` integer, -- total rebounds. empty or null refers to none total rebounds，. 3: totally gets 3 rebounds including offence and defense rebounds. commensense evidence:. total rebounds = offense rebounds + defense rebounds.
    `assists` integer, -- assistants. null or empty refers to none. 2: 2 assistants.
    `steals` integer, -- steals. null or empty refers to none. 2: 2 steals.
    `blocks` integer, -- blocks. null or empty refers to none. 2: 2 blocks.
    `turnovers` integer, -- turnovers. null or empty refers to none. 2: 2 turnovers.
    `personal_fouls` integer, -- personal fouls. null or empty refers to none. 2: 2 personal fouls.
    `fg_attempted` integer, -- field goal attempted. null or empty refers to none. 2: 2 field goal attempts.
    `fg_made` integer, -- field goal made. null or empty refers to none. 2: 2 field goal made.
    `ft_attempted` integer, -- free throw attempted. null or empty refers to none. 2: 2 free throw attempts.
    `ft_made` integer, -- free throw made. null or empty refers to none. 2: 2 free throw made.
    `three_attempted` integer, -- three point attempted. null or empty refers to none. 2: 2 three point attempts.
    `three_made` integer, -- three point made. null or empty refers to none. 2: 2 three point made.
    primary key (playerID, season_id),
    foreign key (playerID) references players (playerID)
);

