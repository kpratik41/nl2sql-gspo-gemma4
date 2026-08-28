CREATE TABLE movies (
    `movieid` integer, -- movie id. unique identifier number of movies.
    `year` integer, -- 4: newest; 1: oldest. higher value means newer published date.
    `isEnglish` text, -- is English.
    `country` text, -- country.
    `runningtime` integer, -- No description.
    primary key (movieid)
);

CREATE TABLE u2base (
    `userid` integer, -- user id. identifier number of users.
    `movieid` integer, -- movie id. identifier number of movie.
    `rating` text, -- ratings of movies. higher value refers to higher satisfactory, each value is the rating of movies left by users.
    primary key (userid, movieid)
);

CREATE TABLE movies2directors (
    `movieid` integer, -- movie id. identifier number of movies.
    `directorid` integer, -- director id. identifier number of directors.
    `genre` text, -- genre of movies.
    primary key (movieid, directorid)
);

CREATE TABLE actors (
    `actorid` integer, -- actor id. unique identificator number of actors.
    `a_gender` text, -- actor gender. M: male; F: female.
    `a_quality` integer, -- actor quality. higher is better, lower is the worse.
    primary key (actorid)
);

CREATE TABLE movies2actors (
    `movieid` integer, -- movie id. identifier number of movies.
    `actorid` integer, -- actor id. identifier number of actors.
    `cast_num` integer, -- cast number.
    primary key (movieid, actorid)
);

CREATE TABLE directors (
    `directorid` integer, -- director id. unique identification number of actors directors.
    `d_quality` integer, -- director quality. higher value is better, lower is the worse.
    `avg_revenue` integer, -- average revenue. higher value is the higher, lower is the lower.
    primary key (directorid)
);

CREATE TABLE users (
    `userid` integer, -- user id. unique identifier number of users.
    `age` text, -- 1: 1-18 years old; 18: 18-25 years old; 25: 25-35 years old; 35: 35-45 years old; 45: 45-50 years old; 50: 50-56 years old; 56: over 56 years old.
    `u_gender` text, -- user gender. M / F: Male / Female.
    `occupation` text, -- occupation.
    primary key (userid)
);

