CREATE TABLE genre (
    `genre_id` integer, -- genre id. the unique identifier of the genre.
    `genre_name` text, -- the genre.
    primary key (genre_id)
);

CREATE TABLE movie (
    `movie_id` integer, -- movie id. the unique identifier of the movie.
    `title` text, -- the title of the movie.
    `budget` integer, -- the budget for the movie. If a movie has higher popularity, it means that it is well-liked by a large number of people. This can be determined by looking at the movie's ratings and reviews, as well as the box office performance and overall buzz surrounding the film. Higher popularity often translates to more success for the movie, both financially and critically.
    `homepage` text, -- the homepage of the movie.
    `overview` text, -- the overview of the movie.
    `popularity` real, -- the popularity of the movie. If a movie has higher popularity, it means that it is well-liked by a large number of people. This can be determined by looking at the movie's ratings and reviews, as well as the box office performance and overall buzz surrounding the film. Higher popularity often translates to more success for the movie, both financially and critically.
    `release_date` date, -- release date. the release date of the movie.
    `revenue` integer, -- the revenue of the movie. A higher vote average indicates that a greater proportion of people who have seen the movie have given it positive ratings.
    `runtime` integer, -- the runtime of the movie.
    `movie_status` text, -- the status of the movie. The only value of this column is 'Released'.
    `tagline` text, -- the tagline of the movie.
    `vote_average` real, -- vote average. the average vote for the movie. A higher vote average indicates that a greater proportion of people who have seen the movie have given it positive ratings.
    `vote_count` integer, -- vote count. the vote count for the movie. If a movie has a higher vote average and vote count, it means that it has been well-received by audiences and critics. A higher vote count means that more people have rated the movie, which can indicate a greater level of interest in the film.
    primary key (movie_id)
);

CREATE TABLE movie_cast (
    `movie_id` integer, -- movie id. the id of the movie. Maps to movie(movie_id).
    `person_id` integer, -- person id. the id of the person. Maps to person(person_id).
    `character_name` text, -- character name. the character name.
    `gender_id` integer, -- gender id. the id of the cast's gender. Maps to gender(gender_id).
    `cast_order` integer, -- cast order. the cast order of the cast. The cast order of a movie or television show refers to the sequence in which the actors and actresses are listed in the credits. This order is typically determined by the relative importance of each actor's role in the production, with the main actors and actresses appearing first, followed by the supporting cast and extras.
    foreign key (gender_id) references gender(gender_id),
    foreign key (movie_id) references movie(movie_id),
    foreign key (person_id) references person(person_id)
);

CREATE TABLE person (
    `person_id` integer, -- person id. the unique identifier of the person.
    `person_name` text, -- person name. the name of the person.
    primary key (person_id)
);

CREATE TABLE department (
    `department_id` integer, -- department id. the unique identifier of the department.
    `department_name` text, -- department name. the name of the department.
    primary key (department_id)
);

CREATE TABLE gender (
    `gender_id` integer, -- gender id. the unique identifier of the gender.
    `gender` text, -- the gender. female/ male/ unspecified.
    primary key (gender_id)
);

CREATE TABLE keyword (
    `keyword_id` integer, -- keyword id. the unique identifier of the keyword.
    `keyword_name` text, -- keyword name. the keyword.
    primary key (keyword_id)
);

CREATE TABLE movie_keywords (
    `movie_id` integer, -- movie id. the id of the movie. Maps to movie(movie_id).
    `keyword_id` integer, -- keyword id. the id of the movie keyword. Maps to keyword(keyword_id). A movie may have many keywords. Audience could get the genre of the movie according to the movie keywords.

);

CREATE TABLE movie_languages (
    `movie_id` integer, -- movie id. the id of the movie. Maps to movie(movie_id).
    `language_id` integer, -- language id. the id of the movie language. Maps to language(language_id).
    `language_role_id` integer, -- language role id. the id of the role's language.
    foreign key (language_id) references language(language_id),
    foreign key (movie_id) references movie(movie_id),
    foreign key (language_role_id) references language_role(role_id)
);

CREATE TABLE movie_genres (
    `movie_id` integer, -- movie id. the id of the movie. Maps to movie(movie_id).
    `genre_id` integer, -- genre id. the id of the movie genre. Maps to genre(genre_id).
    foreign key (genre_id) references genre(genre_id),
    foreign key (movie_id) references movie(movie_id)
);

CREATE TABLE country (
    `country_id` integer, -- country id. the unique identifier of the country.
    `country_iso_code` text, -- country iso code. the ISO code. ISO codes are typically used to identify countries and their subdivisions, and there are different types of ISO codes depending on the specific application. Here we use ISO 3166 code to identify countries.
    `country_name` text, -- country name. the name of the country.
    primary key (country_id)
);

CREATE TABLE language_role (
    `role_id` integer, -- role id. the unique identifier of the language id.
    `language_role` text, -- language role. the language role. In the context of language roles in a movie or other audio-visual production, "original" and "spoken" refer to the languages in which the movie was originally produced, and the languages spoken by the characters in the movie, respectively.
    primary key (role_id)
);

CREATE TABLE movie_company (
    `movie_id` integer, -- movie id. the id of the movie. Maps to movie(movie_id).
    `company_id` integer, -- company id. the id of the company that produced the movie. Maps to production_company(company_id). If movies with different movie_id have the same company_id, it means these movies were made by the same company.

);

CREATE TABLE movie_crew (
    `movie_id` integer, -- movie id. the id of the movie that the crew worked for. Maps to movie(movie_id).
    `person_id` integer, -- person id. the id of the crew. Maps to person(person_id).
    `department_id` integer, -- department id. the id of the crew's department. Maps to department(department_id).
    `job` text, -- the job of the crew. A movie may involve several crews with the same job title.
    foreign key (department_id) references department(department_id),
    foreign key (movie_id) references movie(movie_id),
    foreign key (person_id) references person(person_id)
);

CREATE TABLE production_country (
    `movie_id` integer, -- mivie id. the unique identifier of the movie.
    `country_id` integer, -- country id. the id of the country.
    foreign key (country_id) references country(country_id),
    foreign key (movie_id) references movie(movie_id)
);

CREATE TABLE language (
    `language_id` integer, -- language id. the unique identifier of the language.
    `language_code` text, -- language code. the code of the language. Here we use ISO 639 codes to identify the language.
    `language_name` text, -- language name. the language name.
    primary key (language_id)
);

CREATE TABLE production_company (
    `company_id` integer, -- company id. the unique identifier of the company.
    `company_name` text, -- company name. the name of the company.
    primary key (company_id)
);

