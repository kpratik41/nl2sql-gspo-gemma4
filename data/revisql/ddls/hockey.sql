CREATE TABLE GoaliesSC (
    `playerID` text, -- id number identifying the player. number identifying players.
    `year` integer, -- year. 2005-06 listed as "2005".
    `tmID` text, -- team ID. team abbreviated name.
    `lgID` text, -- league ID. league abbreviated name.
    `GP` integer, -- Games played.
    `Min` integer, -- Minutes. Minutes of appearance.
    `W` integer, -- Wins.
    `L` integer, -- Loses.
    `T` integer, -- Ties.
    `SHO` integer, -- Shutouts. In ice hockey, a shutout (SHO) is credited to a goaltender who successfully stops the other team from scoring during the entire game.
    `GA` integer, -- Goals against. �Goals Against� are the number of goals recorded while the goalie is on the ice. Include all goals against during regulation and overtime play.
    primary key (playerID, year),
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE ScoringSup (
    `playerID` text, -- player id. string ID of players. unique string ID of the player.
    `year` integer, -- No description.
    `PPA` text, -- Power play assists.
    `SHA` text, -- Shorthanded assists.
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE Coaches (
    `coachID` text, -- coach ID. number identifying coaches.
    `year` integer, -- In which year did the coach teach this team.
    `tmID` text, -- team ID. team abbreviated name.
    `lgID` text, -- league ID. league abbreviated name.
    `stint` integer, -- stint.
    `notes` text, -- note if needed. noted information.
    `g` integer, -- games. number of games.
    `w` integer, -- wins. number of wins.
    `l` integer, -- loses. number of loses.
    `t` integer, -- ties. number of ties.
    `postg` text, -- post-season games. number of post-season games.
    `postw` text, -- post-season wins. number of post-season wins.
    `postl` text, -- post-season loses. number of post-season loses.
    `postt` text, -- post-season ties. number of post-season ties.
    primary key (coachID, year, tmID, stint),
    foreign key (year, tmID) references Teams (year, tmID)
);

CREATE TABLE Teams (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `franchID` text, -- Franchise ID.
    `confID` text, -- Conference ID. see abbrev.csv for details.
    `divID` text, -- Division ID. see abbrev.csv for details.
    `rank` integer, -- Final standing.
    `playoff` text, -- playoff results.
    `G` integer, -- games.
    `W` integer, -- wins.
    `L` integer, -- loses.
    `T` integer, -- ties.
    `OTL` text, -- Overtime losses.
    `Pts` integer, -- points.
    `SoW` text, -- Shootout wins.
    `SoL` text, -- Shootout loses.
    `GF` integer, -- Goals for.
    `GA` integer, -- Goals against. goals against.
    `name` text, -- Full team name.
    `PIM` text, -- Penalty minutes.
    `BenchMinor` text, -- Bench minors (minutes). A bench minor penalty is a minor penalty committed by a player or coach that is not on the ice. It is like a minor penalty in that it calls for the offending player to serve two minutes in the penalty box.
    `PPG` text, -- Power play goals.
    `PPC` text, -- Power play chances. power play percentage (PP%) = PPG / PPC.
    `SHA` text, -- Shorthanded goals against.
    `PKG` text, -- Power play goals against.
    `PKC` text, -- Penalty kill chances.
    `SHF` text, -- Shorthanded goals for.
    primary key (year, tmID)
);

CREATE TABLE AwardsCoaches (
    `coachID` text, -- coach ID. string ID of the coach.
    `award` text, -- awards that the coach achieve.
    `year` integer, -- year of award.
    `lgID` text, -- league ID. league abbreviated name.
    `note` text, -- No description.
    foreign key (coachID) references Coaches (coachID)
);

CREATE TABLE AwardsMisc (
    `name` text, -- unique name of awards.
    `ID` text, -- id number of players or coaches, etc.
    `award` text, -- awarder. not useful.
    `year` integer, -- year of the award.
    `lgID` text, -- league ID. league abbreviated name.
    `note` text, -- note if needed. noted information.
    primary key (name)
);

CREATE TABLE TeamsHalf (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `half` integer, -- First or second half of season.
    `rank` integer, -- Final standing for the half.
    `G` integer, -- Games.
    `W` integer, -- wins.
    `L` integer, -- loses.
    `T` integer, -- ties.
    `GF` integer, -- goals for.
    `GA` integer, -- goals against. goal against.
    primary key (year, tmID, half),
    foreign key (tmID, year) references Teams (tmID, year)
);

CREATE TABLE abbrev (
    `Type` text, -- type of hockey games.
    `Code` text, -- abbreviated codes.
    `Fullname` text, -- full names of code.
    primary key (Type, Code)
);

CREATE TABLE ScoringSC (
    `playerID` text, -- id number identifying the player. number identifying the player.
    `year` integer, -- year. 2005-06 listed as "2005".
    `tmID` text, -- team ID. team abbreviated name.
    `lgID` text, -- league ID. league abbreviated name.
    `pos` text, -- position.
    `GP` integer, -- Games played.
    `G` integer, -- Goals. goals.
    `A` integer, -- assists.
    `Pts` integer, -- points.
    `PIM` integer, -- Penalty minutes.
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE CombinedShutouts (
    `year` integer, -- year.
    `month` integer, -- month.
    `date` integer, -- day. the entire date in this table is "year" / "month" / "date".
    `tmID` text, -- team ID. team abbreviated name.
    `oppID` text, -- opposite team ID. Team ID of opponent.
    `R/P` text, -- regular / postseason. R" for regular season, or "P" for postseason.
    `IDgoalie1` text, -- ID of goalie 1. ID of first goalie.
    `IDgoalie2` text, -- ID of goalie 2. ID of second goalie.
    foreign key (IDgoalie1) references Master (playerID)
    foreign key (IDgoalie2) references  Master (playerID)
);

CREATE TABLE TeamSplits (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `hW` integer, -- home wins.
    `hL` integer, -- home loses.
    `hT` integer, -- home ties.
    `hOTL` text, -- Home overtime losses.
    `rW` integer, -- Road wins.
    `rL` integer, -- Road loses.
    `rT` integer, -- Road ties.
    `rOTL` text, -- road overtime loses.
    `SepW` text, -- September wins.
    `SepL` text, -- September loses.
    `SepT` text, -- September ties.
    `SepOL` text, -- September overtime loses.
    `OctW` text, -- October wins.
    `OctL` text, -- October loses.
    `OctT` text, -- October ties.
    `OctOL` text, -- October overtime loses.
    `NovW` text, -- November wins.
    `NovL` text, -- November loses.
    `NovT` text, -- November ties.
    `NovOL` text, -- November overtime loses.
    `DecW` text, -- December wins.
    `DecL` text, -- December loses.
    `DecT` text, -- December ties.
    `DecOL` text, -- December overtime loses.
    `JanW` integer, -- January wins.
    `JanL` integer, -- January loses.
    `JanT` integer, -- January ties.
    `JanOL` text, -- January overtime loses.
    `FebW` integer, -- February wins.
    `FebL` integer, -- February loses.
    `FebT` integer, -- February ties.
    `FebOL` text, -- February overtime loses.
    `MarW` text, -- March wins.
    `MarL` text, -- March loses.
    `MarT` text, -- March ties.
    `MarOL` text, -- March overtime loses.
    `AprW` text, -- April wins.
    `AprL` text, -- April loses.
    `AprT` text, -- April ties.
    `AprOL` text, -- April overtime loses.
    primary key (year, tmID),
    foreign key (year, tmID) references Teams (year, tmID)
);

CREATE TABLE SeriesPost (
    `year` integer, -- year.
    `round` text, -- round. see abbrev.csv for details.
    `series` text, -- series. used for the NHL designations.
    `tmIDWinner` text, -- Team ID of winner.
    `lgIDWinner` text, -- League ID of winner.
    `tmIDLoser` text, -- Team ID of loser. Team ID of winner.
    `lgIDLoser` text, -- league id of loser.
    `W` integer, -- number of wins of the winner.
    `L` integer, -- number of loses of the winner.
    `T` integer, -- number of ties.
    `GoalsWinner` integer, -- goals for winner.
    `GoalsLoser` integer, -- goals for loser.
    `note` note, -- note. Note: EX = exhibition, ND = no decision, TG = total-goals series.
    foreign key (year, tmIDWinner) references Teams (year, tmID)
    foreign key (year, tmIDLoser) references Teams (year, tmID)
);

CREATE TABLE Master (
    `playerID` text, -- id number identifying the player.
    `coachID` text, -- coach ID. coach id number. if a person has both playerID and coachID, it means this person becomes coach after retirement.
    `hofID` text, -- hall of fame id.
    `firstName` text, -- first name.
    `lastName` text, -- last name.
    `nameNote` text, -- name note. note about name.
    `nameGiven` text, -- name given. Given name.
    `nameNick` text, -- Nickname. (multiple entries separated by "/").
    `height` text, -- height.
    `weight` text, -- weight.
    `shootCatch` text, -- shoot catch. Shooting hand (or catching hand for goalies). Shooting hand (or catching hand for goalies). L: left hand. R: right hand. if null or 'empty', it means this player is good at both left and right hand.
    `legendsID` text, -- legends ID. ID at Legends Of Hockey.
    `ihdbID` text, -- Internet Hockey Database ID. ID at the Internet Hockey Database.
    `hrefID` text, -- Hockey-Reference.com ID. ID at Hockey-Reference.com.
    `firstNHL` text, -- First NHL season.
    `lastNHL` text, -- Last NHL season.
    `firstWHA` text, -- First WHA season.
    `lastWHA` text, -- Last WHA season.
    `pos` text, -- position. LW: left winger. RW: right winger. C: center. G: goalie. D: defenseman. W: winger. F: forward. the player with W (winger) means he can play the role as both LW (left winger) and RW (right winger). some players have two positions, which will be shown as "L/D". It means that LW + D --> left winger and defenseman.
    `birthYear` text, -- birth Year.
    `birthMon` text, -- birth Month.
    `birthDay` text, -- birth Day. the entire / detail birth date in this table is "year" / "month" / "date".
    `birthCountry` text, -- birth Country.
    `birthState` text, -- birth State.
    `birthCity` text, -- birth city.
    `deathYear` text, -- death year.
    `deathMon` text, -- death month.
    `deathDay` text, -- death day.
    `deathCountry` text, -- death country.
    `deathState` text, -- death state.
    `deathCity` text, -- death city.
    foreign key (coachID) references Coaches (coachID)
);

CREATE TABLE TeamsPost (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `G` integer, -- Games.
    `W` integer, -- wins.
    `L` integer, -- loses.
    `T` integer, -- ties.
    `GF` integer, -- goals for.
    `GA` integer, -- goals against.
    `PIM` text, -- penalty minutes.
    `BenchMinor` text, -- Bench minors (minutes).
    `PPG` text, -- Power play goals.
    `PPC` text, -- Power play chances.
    `SHA` text, -- Shorthanded goals against.
    `PKG` text, -- Power play goals against.
    `PKC` text, -- Penalty kill chances.
    `SHF` text, -- Shorthanded goals for.
    primary key (year, tmID),
    foreign key (year, tmID) references Teams (year, tmID)
);

CREATE TABLE AwardsPlayers (
    `playerID` text, -- player id. string ID of players. unique string ID of the player.
    `award` text, -- award name.
    `year` integer, -- year of award.
    `lgID` text, -- league ID. league abbreviated name.
    `note` text, -- note if needed.
    `pos` text, -- position. position of players. LW: left winger. RW: right winger. C: center. G: goalie. D: defenseman. W: winger. F: forward. the player with W (winger) means he can play the role as both LW (left winger) and RW (right winger).
    primary key (playerID, award, year),
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE HOF (
    `year` integer, -- year of hall of fame.
    `hofID` text, -- hall of fame id.
    `name` text, -- name.
    `category` text, -- category.
    primary key (hofID)
);

CREATE TABLE GoaliesShootout (
    `playerID` integer, -- No description.
    `year` integer, -- year. 2005-06 listed as "2005".
    `stint` text, -- stint.
    `tmID` integer, -- team ID. team abbreviated name.
    `W` integer, -- Wins.
    `L` integer, -- Loses.
    `SA` integer, -- Shots against. �Shot Against� are the number of shots recorded while the goalie is on the ice.
    `GA` integer, -- Goals against. �Goals Against� are the number of goals recorded while the goalie is on the ice. Include all goals against during regulation and overtime play.
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID)  references Master (playerID)
);

CREATE TABLE TeamVsTeam (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `oppID` text, -- opponent ID.
    `W` integer, -- wins.
    `L` integer, -- loses.
    `T` integer, -- ties.
    `OTL` text, -- overtime loses.
    primary key (year, tmID, oppID),
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (oppID, year) references Teams (tmID, year)
);

CREATE TABLE Scoring (
    `playerID` text, -- player ID.
    `year` integer, -- play year.
    `stint` integer, -- Stint (order of appearance in a season).
    `tmID` text, -- team id.
    `lgID` text, -- league id.
    `pos` text, -- position. LW: left winger. RW: right winger. C: center. G: goalie. D: defenseman. W: winger. F: forward. the player with W (winger) means he can play the role as both LW (left winger) and RW (right winger). some players have two positions, which will be shown as "L/D". It means that LW + D --> left winger and defenseman.
    `GP` integer, -- game played.
    `G` integer, -- goals.
    `A` integer, -- assists.
    `Pts` integer, -- points.
    `PIM` integer, -- Penalty minutes.
    `+/-` text, -- Plus / minus. The plus minus stat is used to determine how often a player is on the ice when a goal is scored for the team versus against the team. A positive plus minus means that the player has been on for more goals scored than against, while a negative number means they have been on for more against. In another words, higher "+ / -" means more importance to the team, lower means less importance.
    `PPG` text, -- Power play goals. When a team with more players on the ice scores a goal it is known as a power play goal. Many goals are scored in power play situations as the team is able to keep possession of the puck and have more scoring chances.
    `PPA` text, -- Power play assists.
    `SHG` text, -- Shorthanded goals. Sometimes the team with fewer players on the ice known as the short-handed team will score a goal.
    `SHA` text, -- Shorthanded assists.
    `GWG` text, -- Game-winning goals. A game-winning goal (GWG) is the goal scored to put the winning team in excess of the losing team's final score. if a player gets more GWG, it means this player is more trustworthy in the critical moment.
    `GTG` text, -- Game-tying goals. A game-tying goal (GWG) is the goal scored to put the winning team in the ties of the losing team's final score.
    `SOG` text, -- Shots on goal. a shot that enters the goal or would have entered the goal if it had not been blocked by the goalkeeper or another defensive player.
    `PostGP` text, -- Postseason games played.
    `PostG` text, -- Postseason goals.
    `PostA` text, -- Postseason assists.
    `PostPts` text, -- Postseason points.
    `PostPIM` text, -- Postseason penalty minutes.
    `Post+/-` text, -- Postseason Plus / minus.
    `PostPPG` text, -- Postseason power play goals.
    `PostPPA` text, -- Postseason power play assists.
    `PostSHG` text, -- Postseason Shorthanded goals.
    `PostSHA` text, -- Postseason Shorthanded assists.
    `PostGWG` text, -- Postseason game-winning goals.
    `PostSOG` text, -- Postseason shots on goal.
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE TeamsSC (
    `year` integer, -- year.
    `lgID` text, -- league ID. league ID number.
    `tmID` text, -- team ID.
    `G` integer, -- Games.
    `W` integer, -- wins.
    `L` integer, -- loses.
    `T` integer, -- ties.
    `GF` integer, -- goals for.
    `GA` integer, -- goals against.
    `PIM` text, -- penalty minutes.
    primary key (year, tmID),
    foreign key (year, tmID) references Teams (year, tmID)
);

CREATE TABLE ScoringShootout (
    `playerID` text, -- player id. id number identifying the player. number identifying the player.
    `year` integer, -- year. 2005-06 listed as "2005".
    `stint` integer, -- stint.
    `tmID` text, -- team ID. team abbreviated name.
    `S` integer, -- shots.
    `G` integer, -- goals.
    `GDG` integer, -- game deciding goals.
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID) references Master (playerID)
);

CREATE TABLE Goalies (
    `playerID` text, -- id number identifying the player. number identifying players.
    `year` integer, -- year. 2005-06 listed as "2005".
    `stint` integer, -- order of appearance in a season. the entire date in this table is "year" / "month" / "date".
    `tmID` text, -- team ID. team abbreviated name.
    `lgID` text, -- league ID. league abbreviated name.
    `GP` text, -- Games played.
    `Min` text, -- Minutes. Minutes of appearance.
    `W` text, -- wins.
    `L` text, -- loses.
    `T/OL` text, -- Ties / overtime losses.
    `ENG` text, -- Empty net goals. An empty net goal happens when a team scores against their opponent who has pulled their goalie. Since there is no goalie in the net, the net is considered �empty".
    `SHO` text, -- Shutouts. In ice hockey, a shutout (SHO) is credited to a goaltender who successfully stops the other team from scoring during the entire game.
    `GA` text, -- Goals against. �Goals Against� are the number of goals recorded while the goalie is on the ice. Include all goals against during regulation and overtime play.
    `SA` text, -- Shots against. �Shot Against� are the number of shots recorded while the goalie is on the ice.
    `PostGP` text, -- Postseason games played.
    `PostMin` text, -- Postseason minutes.
    `PostW` text, -- Postseason wins.
    `PostL` text, -- Postseason loses.
    `PostT` text, -- Postseason ties.
    `PostENG` text, -- Postseason empty net goals. An empty net goal happens when a team scores against their opponent who has pulled their goalie. Since there is no goalie in the net, the net is considered �empty".
    `PostSHO` text, -- Postseason Shutouts. In ice hockey, a shutout (SHO) is credited to a goaltender who successfully stops the other team from scoring during the entire game.
    `PostGA` text, -- Postseason Goals against. �Goals Against� are the number of goals recorded while the goalie is on the ice. Include all goals against during regulation and overtime play.
    `PostSA` text, -- Postseason Shots against. �Shot Against� are the number of shots recorded while the goalie is on the ice.
    primary key (playerID, year, stint),
    foreign key (year, tmID) references Teams (year, tmID)
    foreign key (playerID) references Master (playerID)
);

