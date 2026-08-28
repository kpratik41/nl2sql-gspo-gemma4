CREATE TABLE movie (
    `MovieID` integer, -- movie id. the unique id for the movie.
    `Title` text, -- the title of the movie.
    `MPAA Rating` text, -- motion picture association of america rating. MPAA rating of the movie. MPAA rating is the movie rating for parents to use as a guide to determine the appropriateness of a film's content for children and teenagers.  rated G: General audiences  All ages admitted.  rated PG: Parental guidance suggested  Some material may not be suitable for pre-teenagers.  rated R: Restricted  Under 17 requires accompanying parent or adult guardian.  rated X: No one under 17 admitted.
    `Budget` integer, -- the budget of the movie. the unit is dollar.
    `Gross` integer, -- the gross of the movie.
    `Release Date` text, -- release date. yyyy-mm-dd.
    `Genre` text, -- the genre of the movie.
    `Runtime` integer, -- the runtime of the movie.
    `Rating` real, -- the rating of the movie. 0.0 - 10.0. Higher ratings mean higher quality and better response.
    `Rating Count` integer, -- rating count. the number of the ratings.
    `Summary` text, -- the summary of the movie.
    primary key (MovieID)
);

CREATE TABLE characters (
    `MovieID` integer, -- movie id. the unique id for the movie.
    `ActorID` integer, -- actor id. the unique id for the actor.
    `Character Name` text, -- character name. the name of the character.
    `creditOrder` integer, -- credit order. order of the character in the credit list.
    `pay` text, -- the salary of the character.
    `screentime` text, -- the screentime of the character. Screentime is directly proportional to the importance of the characters.
    primary key (MovieID, ActorID),
    foreign key (ActorID) references actor(ActorID),
    foreign key (MovieID) references movie(MovieID)
);

CREATE TABLE actor (
    `ActorID` integer, -- actor id. the unique id for the actor.
    `Name` text, -- actor's name.
    `Date of Birth` date, -- date of birth. actor's birth date.
    `Birth City` text, -- birth city. actor's birth city.
    `Birth Country` text, -- birth country. actor's birth country.
    `Height (Inches)` integer, -- height inches. actor's height. the unit is inch.
    `Biography` text, -- actor's biography.
    `Gender` text, -- actor's gender.
    `Ethnicity` text, -- actor's ethnicity.
    `NetWorth` text, -- actor's networth. The actor with more networth is richer.
    primary key (ActorID)
);

