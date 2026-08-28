CREATE TABLE Character_Award (
    `award_id` integer, -- award id. A unique identifier for the award.
    `character` text, -- the name of the awarded character.
    foreign key (award_id) references Award(award_id)
);

CREATE TABLE Vote (
    `episode_id` text, -- episode id. A unique identifier for episodes.
    `stars` integer, -- the star score of the episode. 1-10. Star classification is a type of rating scale. The lowest star is 1 which means the worst, and the highest star is 10 which means the best.
    `votes` integer, -- the number of votes of the star score.
    `percent` real, -- the percent of the vote number of the star score. percent = the number of votes for the star / the total amount of votes for all stars.
    foreign key (episode_id) references Episode(episode_id)
);

CREATE TABLE Keyword (
    `episode_id` text, -- episode id. A unique identifier for episodes.
    `keyword` text, -- the keywords of episode.
    primary key (episode_id, keyword),
    foreign key (episode_id) references Episode(episode_id)
);

CREATE TABLE Credit (
    `episode_id` text, -- episode id. A unique identifier for episodes.
    `category` text, -- the category of the credit.
    `person` text, -- the name of cast and crew members.
    `role` text, -- the role of the person.
    `credited` text, -- whether the person is credited. true/ false.  true: The person is included in the credit list.  false: The person isn't included in the credit list.
    foreign key (episode_id) references Episode(episode_id),
    foreign key (person) references Person(name)
);

CREATE TABLE Episode (
    `episode_id` text, -- episode id. A unique identifier for episodes.
    `season` integer, -- the season of the episode.
    `episode` integer, -- the episode number of the episode.
    `number_in_series` integer, -- number in series. the number in series.
    `title` text, -- the title of the episode.
    `summary` text, -- the summary of the episode.
    `air_date` text, -- air date. the air date of the episode. YYYY-MM-DD.
    `episode_image` text, -- episode image. the image of episode.
    `rating` real, -- the rating of episode. 0.0 - 10.0. Higher ratings mean higher quality and better response.  excellent: 7.0 < rating <= 10.0.  average: 5.0 < rating <= 7.0.  bad: 0.0 < rating <= 5.0. not bad: average, excellent.
    `votes` integer, -- the votes of episode. Higher votes mean more audience supports (or popular).
    primary key (episode_id)
);

CREATE TABLE Award (
    `award_id` integer, -- award id. the unique id for the award.
    `organization` text, -- the organization that holds the award.
    `year` integer, -- year of award.
    `award_category` text, -- the category of the award.
    `award` text, -- the name of the award.
    `person` text, -- the person who gets the award.
    `role` text, -- the role of the honoree.
    `episode_id` text, -- episode id. S stands for 'Season' and E stands for 'Episode'.
    `season` text, -- the season of the awarded work.
    `song` text, -- the theme song of the awarded work.
    `result` text, -- the final award result.  Nominee: the prospective recipient of the award. The nominee are people who were nominated but didn't win the award.  Winner: the people who finally won the award.
    primary key (award_id),
    foreign key (person) references Person(name),
    foreign key (episode_id) references Episode(episode_id)
);

CREATE TABLE Person (
    `name` text, -- the name of the crew.
    `birthdate` text, -- birth date. the birth date of the crew. YYYY-MM-DD.
    `birth_name` text, -- birth name. the birth name of the crew.
    `birth_place` text, -- birth place. the birth place of the crew.
    `birth_region` text, -- birth region. the birth region of the crew.
    `birth_country` text, -- birth country. the birth country of the crew.
    `height_meters` real, -- height meters. the height of the crew. the unit is meter.
    `nickname` text, -- the nickname of the crew.
    primary key (name)
);

