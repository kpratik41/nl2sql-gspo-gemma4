CREATE TABLE Bowler_Scores (
    MatchID int NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    GameNumber smallint NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    BowlerID int NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    RawScore smallint NULL DEFAULT 0, -- example: [146, 166, 140]
    HandiCapScore smallint NULL DEFAULT 0, -- example: [192, 205, 171]
    WonGame BOOLEAN NOT NULL DEFAULT 0, -- example: [0, 1]
    PRIMARY KEY (MatchID, GameNumber, BowlerID),
    FOREIGN KEY (BowlerID) REFERENCES Bowlers(BowlerID),
    FOREIGN KEY (MatchID, GameNumber) REFERENCES Match_Games(MatchID, GameNumber)
)

CREATE TABLE Bowler_Scores_Archive (
    MatchID int NOT NULL DEFAULT 0, -- example: NULL
    GameNumber smallint NOT NULL DEFAULT 0, -- example: NULL
    BowlerID int NOT NULL DEFAULT 0, -- example: NULL
    RawScore smallint NULL DEFAULT 0, -- example: NULL
    HandiCapScore smallint NULL DEFAULT 0, -- example: NULL
    WonGame BOOLEAN NOT NULL DEFAULT 0, -- example: NULL
    PRIMARY KEY (MatchID, GameNumber, BowlerID),
    FOREIGN KEY (MatchID, GameNumber) REFERENCES Match_Games_Archive(MatchID, GameNumber)
)

CREATE TABLE Bowlers (
    BowlerID INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
    BowlerLastName TEXT NULL, -- example: ['Black', 'Clothier', 'Cunningham']
    BowlerFirstName TEXT NULL, -- example: ['Barbara', 'David', 'John']
    BowlerMiddleInit TEXT NULL, -- example: ['A', 'J', 'K']
    BowlerAddress TEXT NULL, -- example: ['67 Willow Drive', '2957 W 33rd', '17950 N 59th']
    BowlerCity TEXT NULL, -- example: ['Bothell', 'Ballard', 'Seattle']
    BowlerState TEXT NULL, -- example: ['WA']
    BowlerZip TEXT NULL, -- example: ['98014', '98154', '98011']
    BowlerPhoneNumber TEXT NULL, -- example: ['(206) 555-9876', '(206) 555-7854', '(206) 555-9893']
    TeamID int NULL, -- example: [1, 2, 3]
    BowlerTotalPins int NULL DEFAULT 0, -- example: [5790, 6152, 6435]
    BowlerGamesBowled int NULL DEFAULT 0, -- example: [39, 0]
    BowlerCurrentAverage smallint NULL DEFAULT 0, -- example: [148, 158, 165]
    BowlerCurrentHcp smallint NULL DEFAULT 0, -- example: [47, 38, 32]
    FOREIGN KEY (TeamID) REFERENCES Teams(TeamID)
)

CREATE TABLE sqlite_sequence(name,seq)

CREATE TABLE Match_Games (
    MatchID int NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    GameNumber smallint NOT NULL DEFAULT 0, -- example: [1, 2, 3]
    WinningTeamID int NULL DEFAULT 0, -- example: [1, 2, 3]
    PRIMARY KEY (MatchID, GameNumber)
)

CREATE TABLE Match_Games_Archive (
    MatchID int NOT NULL DEFAULT 0, -- example: NULL
    GameNumber smallint NOT NULL DEFAULT 0, -- example: NULL
    WinningTeamID int NULL DEFAULT 0, -- example: NULL
    PRIMARY KEY (MatchID, GameNumber)
)

CREATE TABLE Teams (
    TeamID INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
    TeamName TEXT NOT NULL, -- example: ['Marlins', 'Sharks', 'Terrapins']
    CaptainID int NULL -- example: [2, 5, 12]
)

CREATE TABLE Tournaments (
    TourneyID INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
    TourneyDate DATE NULL, -- example: ['2017-09-04', '2017-09-11', '2017-09-18']
    TourneyLocation TEXT NULL -- example: ['Red Rooster Lanes', 'Thunderbird Lanes', 'Bolero Lanes']
)

CREATE TABLE Tournaments_Archive (
    TourneyID int NOT NULL DEFAULT 0, -- example: NULL
    TourneyDate DATE NULL, -- example: NULL
    TourneyLocation TEXT NULL, -- example: NULL
    PRIMARY KEY (TourneyID)
)

CREATE TABLE Tourney_Matches (
    MatchID INTEGER PRIMARY KEY AUTOINCREMENT, -- example: [1, 2, 3]
    TourneyID int NULL DEFAULT 0, -- example: [1, 2, 3]
    Lanes TEXT NULL, -- example: ['01-02', '03-04', '05-06']
    OddLaneTeamID int NULL DEFAULT 0, -- example: [1, 2, 3]
    EvenLaneTeamID int NULL DEFAULT 0, -- example: [1, 2, 3]
    FOREIGN KEY (EvenLaneTeamID) REFERENCES Teams(TeamID),
    FOREIGN KEY (OddLaneTeamID) REFERENCES Teams(TeamID),
    FOREIGN KEY (TourneyID) REFERENCES Tournaments(TourneyID)
)

CREATE TABLE Tourney_Matches_Archive (
    MatchID int NOT NULL DEFAULT 0, -- example: NULL
    TourneyID int NULL DEFAULT 0, -- example: NULL
    Lanes TEXT NULL, -- example: NULL
    OddLaneTeamID int NULL DEFAULT 0, -- example: NULL
    EvenLaneTeamID int NULL DEFAULT 0, -- example: NULL
    PRIMARY KEY (MatchID),
    FOREIGN KEY (TourneyID) REFERENCES Tournaments_Archive(TourneyID)
)

CREATE TABLE WAZips (
    ZIP TEXT NOT NULL, -- example: ['98001', '98002', '98003']
    City TEXT NULL, -- example: ['Auburn', 'Federal Way', 'Bellevue']
    State TEXT NULL, -- example: ['WA']
    PRIMARY KEY (ZIP)
)

