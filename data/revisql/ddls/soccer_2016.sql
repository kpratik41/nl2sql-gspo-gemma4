CREATE TABLE Outcome (
    `Outcome_Id` integer, -- outcome id. unique id for outcome.
    `Outcome_Type` text, -- outcome type. type of the outcome.
    primary key (Outcome_Id)
);

CREATE TABLE Player_Match (
    `Match_Id` integer, -- match id. unique id for match.
    `Player_Id` integer, -- player id. the id of the player.
    `Role_Id` integer, -- role id. the id of the play's role in the match. if a player has multiple roles in a match, it means this player is versatile.
    `Team_Id` integer, -- team id. the id of player's team.
    primary key (Match_Id, Player_Id, Role_Id),
    foreign key (Match_Id) references Match(Match_Id),
    foreign key (Player_Id) references Player(Player_Id),
    foreign key (Team_Id) references Team(Team_Id),
    foreign key (Role_Id) references Rolee(Role_Id)
);

CREATE TABLE Win_By (
    `Win_Id` integer, -- winning id. the unique id for the winning.
    `Win_Type` text, -- winning type. the winning type.
    primary key (Win_Id)
);

CREATE TABLE Match (
    `Match_Id` integer, -- unique id for match.
    `Team_1` integer, -- team 1. the team id of the first team.
    `Team_2` integer, -- team 2. the team id for the second team.
    `Match_Date` date, -- match date. the date of the match. yyyy-mm-dd.
    `Season_Id` integer, -- season id. the id of the season.
    `Venue_Id` integer, -- venue id. the id of the venue where the match held.
    `Toss_Winner` integer, -- toss winner. the id of the toss winner. The toss winner is a term used in cricket and is the team that wins the heads or tails toss of coin at the beginning of a match which then enables the team captain to decide whether to bat or bowl first on the pitch.
    `Toss_Decide` integer, -- toss decide. the decision (batting or bowling) made after winning the toss. â¢ field. â¢ bat.
    `Win_Type` integer, -- winning type. the id of the winning type.
    `Win_Margin` integer, -- winning margin. the points of the winning margin. A winning margin bet is a wager on the final result of a game, within a certain range of points.
    `Outcome_type` integer, -- outcome type. the id of the outcome type.
    `Match_Winner` integer, -- match winner. the team id of the match winner.
    `Man_of_the_Match` integer, -- man of the match. the id of the man of the match. In team sport, a player of the match or man (or woman) of the match award is often given to the outstanding player in a particular match.
    primary key (Match_Id),
    foreign key (Team_1) references Team(Team_Id),
    foreign key (Team_2) references Team(Team_Id),
    foreign key (Season_Id) references Season(Season_Id),
    foreign key (Venue_Id) references Venue(Venue_Id),
    foreign key (Toss_Winner) references Team(Team_Id),
    foreign key (Toss_Decide) references Toss_Decision(Toss_Id),
    foreign key (Win_Type) references Win_By(Win_Id),
    foreign key (Outcome_type) references Out_Type(Out_Id),
    foreign key (Match_Winner) references Team(Team_Id),
    foreign key (Man_of_the_Match) references Player(Player_Id)
);

CREATE TABLE Umpire (
    `Umpire_Id` integer, -- umpire id. the unique id of the umpire.
    `Umpire_Name` text, -- umpire name. umpire's name.
    `Umpire_Country` integer, -- umpire country. the id of the country where the umpire are from.
    primary key (Umpire_Id),
    foreign key (Umpire_Country) references Country(Country_Id)
);

CREATE TABLE Extra_Runs (
    `Match_Id` integer, -- match id. Unique Number Which Identifies a match.
    `Over_Id` integer, -- over id. Unique Number which Identifies an over in an Innings.
    `Ball_Id` integer, -- ball id. Unique Number which Identifies a ball in an over.
    `Extra_Type_Id` integer, -- extra type id. Unique Number which Identifies extra type.
    `Extra_Runs` integer, -- extra runs. Number of extra runs.
    `Innings_No` integer, -- innings number. Unique Number which Identifies an innings in a match.
    primary key (Match_Id, Over_Id, Ball_Id, Innings_No),
    foreign key (Extra_Type_Id) references Extra_Type(Extra_Id)
);

CREATE TABLE Team (
    `Team_Id` integer, -- team id. the unique id for the team.
    `Team_Name` text, -- team name. the team name.
    primary key (Team_Id)
);

CREATE TABLE Batting_Style (
    `Batting_Id` integer, -- batting id. unique id for batting hand.
    `Batting_hand` text, -- batting hand. the batting hand: left or right.
    primary key (Batting_Id)
);

CREATE TABLE Out_Type (
    `Out_Id` integer, -- out id. unique id for out type.
    `Out_Name` text, -- out name. out type name.
    primary key (Out_Id)
);

CREATE TABLE Rolee (
    `Role_Id` integer, -- role id. the unique id for the role.
    `Role_Desc` text, -- role description. the role description.
    primary key (Role_Id)
);

CREATE TABLE Player (
    `Player_Id` integer, -- player id. the id of the player.
    `Player_Name` text, -- player name. the name of the player.
    `DOB` date, -- date of birth. player's birthday. yyyy-mm-dd.
    `Batting_hand` integer, -- batting hand. the id of the batting hand.
    `Bowling_skill` integer, -- bowling skill. the id of the bowling skill.
    `Country_Name` integer, -- country name. the name of the country where the player is from.
    primary key (Player_Id),
    foreign key (Batting_hand) references Batting_Style(Batting_Id),
    foreign key (Bowling_skill) references Bowling_Style(Bowling_Id),
    foreign key (Country_Name) references Country(Country_Id)
);

CREATE TABLE Season (
    `Season_Id` integer, -- season id. the unique id for season.
    `Man_of_the_Series` integer, -- man of the series. the player id of man of the series. In team sport, a player of the series or man (or woman) of the series award is often given to the outstanding player in a particular series.
    `Orange_Cap` integer, -- orange cap. the player id who wins the orange cap. The Orange Cap is a coveted award for a batter who is participating in the Indian Premier League (IPL).
    `Purple_Cap` integer, -- purple cap. the player id who wins the purple cap. The Purple Cap is awarded to the bowler who has emerged as the leading wicket-taker in a particular edition of the high-profile Indian Premier League (IPL).
    `Season_Year` integer, -- season year. the year of the season.
    primary key (Season_Id)
);

CREATE TABLE Bowling_Style (
    `Bowling_Id` integer, -- bowling id. unique id for bowling style.
    `Bowling_skill` text, -- bowling skill. the bowling skill.
    primary key (Bowling_Id)
);

CREATE TABLE Ball_by_Ball (
    `Match_Id` integer, -- match id. Unique Number Which Identifies a match.
    `Over_Id` integer, -- over id. Unique Number which Identifies an over in an Innings. The match is made up of two innings and each team takes a turn at batting and bowling. An innings is made up of 50 overs. An over involves six deliveries from the bowler.
    `Ball_Id` integer, -- ball id. Unique Number which Identifies a ball in an over.
    `Innings_No` integer, -- innings number. Unique Number which Identifies an innings in a match.
    `Team_Batting` integer, -- team batting. Unique Number which Identifies Batting team in a match.
    `Team_Bowling` integer, -- team bowling. Unique Number which Identifies Bowling team in a match.
    `Striker_Batting_Position` integer, -- striker batting position. Unique Number which Identifies the position in which player came into bat.
    `Striker` integer, -- Unique Number which Identifies the player who is on strike for that particular ball.
    `Non_Striker` integer, -- non striker. Unique Number which Identifies the player who is Non-striker for that particular ball.
    `Bowler` integer, -- Unique Number which Identifies the player who is Bowling that particular ball.
    primary key (Match_Id, Over_Id, Ball_Id, Innings_No),
    foreign key (Match_Id) references Match(Match_Id)
);

CREATE TABLE Toss_Decision (
    `Toss_Id` integer, -- toss id. the unique id for the toss.
    `Toss_Name` text, -- toss name. the toss decision name.
    primary key (Toss_Id)
);

CREATE TABLE City (
    `City_Id` integer, -- city id. unique id for city.
    `City_Name` text, -- city name.
    `Country_id` integer, -- country id. id of country.
    primary key (City_Id)
);

CREATE TABLE Extra_Type (
    `Extra_Id` integer, -- extra id. unique id for extra type.
    `Extra_Name` text, -- extra name. extra type name.
    primary key (Extra_Id)
);

CREATE TABLE Batsman_Scored (
    `Match_Id` integer, -- match id. Unique Number Which Identifies a match.
    `Over_Id` integer, -- over id. Unique Number which Identifies an over in an Innings.
    `Ball_Id` integer, -- ball id. Unique Number which Identifies a ball in an over.
    `Runs_Scored` integer, -- runs scored. Number of Runs scored by the batsman.
    `Innings_No` integer, -- innings number. Unique Number which Identifies an innings in a match.
    primary key (Match_Id, Over_Id, Ball_Id, Innings_No),
    foreign key (Match_Id) references Match(Match_Id)
);

CREATE TABLE Country (
    `Country_Id` integer, -- country id. unique id for country.
    `Country_Name` text, -- country name.
    primary key (Country_Id),
    foreign key (Country_Id) references Country(Country_Id)
);

CREATE TABLE Wicket_Taken (
    `Match_Id` integer, -- match id. the id of the match.
    `Over_Id` integer, -- over id. the id of the over in an inning.
    `Ball_Id` integer, -- ball id. the id of the ball in an over.
    `Player_Out` integer, -- player out. the player id who got an out.
    `Kind_Out` integer, -- kind out. the id that represents the out type.
    `Fielders` integer, -- the id of fielders.
    `Innings_No` integer, -- innings number. number which identifies an innings in a match.
    primary key (Match_Id, Over_Id, Ball_Id, Innings_No),
    foreign key (Match_Id) references Match(Match_Id),
    foreign key (Player_Out) references Player(Player_Id),
    foreign key (Kind_Out) references Out_Type(Out_Id),
    foreign key (Fielders) references Player(Player_Id)
);

CREATE TABLE Venue (
    `Venue_Id` integer, -- venue id. the unique id of the venue.
    `Venue_Name` text, -- venue name. the name of the venue.
    `City_Id` integer, -- city id. the city id where the venue is located in.
    primary key (Venue_Id),
    foreign key (City_Id) references City(City_Id)
);

