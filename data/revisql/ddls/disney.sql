CREATE TABLE revenue (
    `Year` integer, -- The year the movie was released.
    `Studio Entertainment[NI 1]` real, -- The studio entertainment segment of the Walt Disney Company.
    `Disney Consumer Products[NI 2]` real, -- The consumer products segment of the Walt Disney Company.
    `Disney Interactive[NI 3][Rev 1]` integer, -- The interactive segment of the Walt Disney Company.
    `Walt Disney Parks and Resorts` real, -- The parks and resorts segment of the Walt Disney Company.
    `Disney Media Networks` text, -- The media networks segment of the Walt Disney Company.
    `Total` integer, -- The total box office gross for the movie.
    primary key (Year)
);

CREATE TABLE voice-actors (
    `character` text, -- The unique name of the character.
    `voice-actor` text, -- The name of the voice actor.
    `movie` text, -- The name of the movie.
    primary key (character),
    foreign key (movie) references characters(movie_title)
);

CREATE TABLE director (
    `name` text, -- unique movie name.
    `director` text, -- the name of the director. one director may have multiple movies. more movies --> this director is more productive.
    primary key (name),
    foreign key (name) references characters(movie_title)
);

CREATE TABLE characters (
    `movie_title` text, -- movie title. unique title of the movie.
    `release_date` text, -- release date. The release date of the movie.
    `hero` text, -- The main character of the movie. round role.
    `villian` text, -- The villain of the movie. a character whose evil actions or motives are important to the plot.
    `song` text, -- A song associated with the movie.
    primary key (movie_title),
    foreign key (hero) references "voice-actors"(character)
);

CREATE TABLE movies_total_gross (
    `movie_title` text, -- movie title.
    `release_date` text, -- release date.
    `genre` text, -- genre of the movie.
    `MPAA_rating` text, -- Motion Picture Association of America rating. Motion Picture Association of America of the disney movie. â¢ G: general audience. â¢ PG: mature audiences or parental guidance suggested. â¢ R: restricted: no children under 17 allowed without parents or adult guardians. â¢ PG-13: PARENTS STRONGLY CAUTIONED. Some material may be inappropriate for children under 13. movies need accompany with parents: PG, PG-13, PG-17; if "Not rated" or null, it means this film can show only gets permissions by theatre management. if the film can show without need of permissions of theatre management, the MPAA_rating should not be "Not rated" or null.
    `total_gross` text, -- total gross. The total gross of the movie. more total_gross--> more popular movie.
    `inflation_adjusted_gross` text, -- inflation adjusted gross. The inflation-adjusted gross of the movie. estimated inflation rate = inflation_adjusted_gross / total_gross; the current gross = inflation_adjusted_gross.
    primary key (movie_title, release_date),
    foreign key (movie_title) references characters(movie_title)
);

