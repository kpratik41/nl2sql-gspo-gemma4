CREATE TABLE noc_region (
    `id` integer, -- the unique identifier of the noc region.
    `noc` text, -- the NOC code of the region. A NOC code is a number used by the International Olympic Committee (IOC) to identify each of the national Olympic committees (NOCs) from around the world.
    `region_name` text, -- region name. the name of the region.
    primary key (id)
);

CREATE TABLE sport (
    `id` integer, -- the unique identifier of the sport.
    `sport_name` text, -- sport name. the name of the sport.
    primary key (id)
);

CREATE TABLE games (
    `id` integer, -- the unique identifier of the game.
    `games_year` integer, -- games year. the year of the game.
    `games_name` text, -- games name. the name of the game. games_name is 'games_year season'.
    `season` text, -- the season of the game. The Olympics are held every four years and include both summer and winter games. The summer games include a wide range of sports, such as athletics, swimming, and gymnastics, while the winter games feature sports that are played on ice or snow, such as ice hockey, figure skating, and skiing.
    primary key (id)
);

CREATE TABLE games_city (
    `games_id` integer, -- games id. the id of the game. Maps to games(id).
    `city_id` integer, -- city id. the id of the city that held the game. Maps to city(id).
    foreign key (city_id) references city(id),
    foreign key (games_id) references games(id)
);

CREATE TABLE person (
    `id` integer, -- the unique identifier of the person.
    `full_name` text, -- full name. the full name of the person. A person's full name is the complete name that they are known by, which typically includes their first name, middle name (if applicable), and last name.
    `gender` text, -- the gender of the person. M stands for male and F stands for female.
    `height` integer, -- the height of the person.
    `weight` integer, -- the weight of the person. The unit of height is cm, and the unit of weight is kg. If the height or the weight of the person is 0, it means his/her height or weight is missing.
    primary key (id)
);

CREATE TABLE event (
    `id` integer, -- the unique identifier of the event.
    `sport_id` integer, -- sport id. the id of the sport. Maps to sport(id). There may be different events for the same sport in Olympics.
    `event_name` text, -- event name. the name of the event.
    primary key (id),
    foreign key (sport_id) references sport(id)
);

CREATE TABLE person_region (
    `person_id` integer, -- the id of the person. Maps to person(id).
    `region_id` integer, -- the id of the noc region. Maps to noc_region(id).
    foreign key (person_id) references person(id),
    foreign key (region_id) references noc_region(id)
);

CREATE TABLE medal (
    `id` integer, -- the unique identifier of the metal.
    `medal_name` text, -- medal name. the name of the medal.
    primary key (id)
);

CREATE TABLE games_competitor (
    `id` integer, -- the unique identifier of the record.
    `games_id` integer, -- games id. the id of the game. Maps to games(id).
    `person_id` integer, -- person id. the id of the person. Maps to person(id). The times a person participated in the Olympics games could be calculated by grouping by person_id.
    `age` integer, -- the age of the person when he/she participated in the game. If person A's age was 22 when he participated in game A and his age was 24 when he participated in game B, it means that game A was held two years earlier than game B.
    primary key (id),
    foreign key (games_id) references games(id),
    foreign key (person_id) references person(id)
);

CREATE TABLE city (
    `id` integer, -- the unique identifier of the city.
    `city_name` text, -- city name. the name of the city.
    primary key (id)
);

CREATE TABLE competitor_event (
    `event_id` integer, -- event id. the id of the event. Maps to event(id).
    `competitor_id` integer, -- competitor id. the id of the competitor. Maps to games_competitor(id). The number of competitors who participated in a specific event could be calculated by grouping by event_id.
    `medal_id` integer, -- medal id. the id of the medal. Maps to medal(id). For a specific event, the competitor whose medal_id = 1 is the champion, the competitor whose medal_id = 2 is the runner-up, and the competitor whose medal_id = 3 got third place.
    foreign key (competitor_id) references games_competitor(id),
    foreign key (event_id) references event(id),
    foreign key (medal_id) references medal(id)
);

