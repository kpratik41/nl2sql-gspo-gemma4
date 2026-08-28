CREATE TABLE Vote (
    `episode_id` text, -- episode id. the id of the episode.
    `stars` integer, -- a number between 1 and 10 indicating how much the viewer enjoyed the episode. The higher the number, the more the episode was enjoyed.
    `votes` integer, -- The total number of viewers that gave the specific episode the number of stars indicated in the "stars" column.
    `percent` real, -- The percent of viewers that gave the specific episode the number of stars indicated in the "stars" column.
    foreign key (episode_id) references Episode(episode_id)
);

CREATE TABLE Keyword (
    `episode_id` text, -- episode id. the id of the episode.
    `keyword` text, -- the keyword that is relevant for the episode.
    primary key (episode_id, keyword),
    foreign key (episode_id) references Episode(episode_id)
);

CREATE TABLE Credit (
    `episode_id` text, -- episode id. the id of the episode to which the credit information pertains.
    `person_id` text, -- person id. the id of the person to which the credit information pertains.
    `category` text, -- the kind of credit being recognized.
    `role` text, -- the role for which the person is being recognized. If the credit is for an actor, this is the name of the character he or she portrayed (Dan Florek plays Donald Craven). Otherwise, this is the production role (producer, sound editor).
    `credited` text, -- whether the credit was displayed in the credits at the end of the episode. A person may fill a role in an episode, but not show up in the on-screen credits. In this case, the work is said to be "uncredited.".
    primary key (episode_id, person_id),
    foreign key (episode_id) references Episode(episode_id),
    foreign key (person_id) references Person(person_id)
);

CREATE TABLE Episode (
    `episode_id` text, -- episode id. the unique identifier of the episode.
    `series` text, -- the name of the series.
    `season` integer, -- a number indicating the season of the episode.
    `episode` integer, -- the sequential number of the episode within a specific season.
    `number_in_series` integer, -- number in series. the overall sequence number of episode in the entire series.
    `title` text, -- the title of the episode.
    `summary` text, -- a brief description of what happens in the episode.
    `air_date` date, -- air date. the date the episode was initially broadcast in the United States.
    `episode_image` text, -- episode image. a link to an image from the episode.
    `rating` real, -- the weighted average of the votes received for the episode. The higher rating means the episode has more positive viewer comments.
    `votes` integer, -- the total number of rating votes for the episode.
    primary key (episode_id)
);

CREATE TABLE Award (
    `award_id` integer, -- award id. the unique identifier for the award nomination.
    `organization` text, -- the name of the organization that grants the award.
    `year` integer, -- the year of the nomination for the award.
    `award_category` text, -- award category. the class of the award.
    `award` text, -- the specific award.
    `series` text, -- the name of the Law & Order series that has been nominated.
    `episode_id` text, -- episode id. the id of the episode that has been nominated.
    `person_id` text, -- the id of the person that has been nominated.
    `role` text, -- the role that has been nominated.
    `result` text, -- the nomination result. Nominee / Winner. 'Winner' means that the award was actually received.
    primary key (award_id),
    foreign key (episode_id) references Episode(episode_id),
    foreign key (person_id) references Person(person_id)
);

CREATE TABLE Person (
    `person_id` text, -- person id. the unique identifier for the person.
    `name` text, -- the name of the person.
    `birthdate` date, -- birth date. the date the person was born. if null, it means this birthdate of the person is not available.
    `birth_name` text, -- birth name. the name parents gave to the person shortly after birth.
    `birth_place` text, -- birth place. the place where a person was born. the birth place info is integrate if birth_name, birth_place, birth_region don't contain null value. the full birth place = birth_country +birth_region +birth_place.
    `birth_region` text, -- birth region. the geographical area describing where a person was born that is larger than the birth place.
    `birth_country` text, -- birth country. the name of the country where the person was born. can ask questions about its corresponding continent: e.g.:. USA --> North America.
    `height_meters` real, -- height meters. how tall the person is in meters.
    `nickname` text, -- the nickname the person goes by.
    primary key (person_id)
);

