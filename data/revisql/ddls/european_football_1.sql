CREATE TABLE matchs (
    `Div` text, -- Division. Division Id.
    `Date` date, -- Match Date. YYYY-MM-DD.
    `HomeTeam` text, -- Name of Home Team.
    `AwayTeam` text, -- Name of Away Team.
    `FTHG` integer, -- Final-time Home-team Goals.
    `FTAG` integer, -- Final-time Away-team Goals.
    `FTR` text, -- Final-time Results. H stands for home victory, which means FTHG is higher than FTAG. A stands for away victory, which means FTAG is higher than FTHG. D stands for draw, which means FTHG equals to FTAG.
    `season` integer, -- season of the match.
    foreign key (Div) references divisions(division)
);

CREATE TABLE divisions (
    `division` text, -- division id.
    `name` text, -- name of the division.
    `country` text, -- country of the division.
    primary key (division)
);

