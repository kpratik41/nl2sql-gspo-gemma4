CREATE TABLE genre (
    `id` integer, -- the unique identifier of the game genre.
    `genre_name` text, -- the game genre. The game genre can have a significant effect on the game. The genre refers to the category or type of game, such as action, adventure, strategy, or role-playing. The genre can determine the general style and gameplay of the game, as well as the expectations of the players. The genre can also affect the audience for the game, as different genres may appeal to different groups of players.
    primary key (id)
);

CREATE TABLE game (
    `id` integer, -- the unique identifier of the game.
    `genre_id` integer, -- genre id. the id of the game genre. If game A and game B have the same genre, the user who likes game A may also like game B.
    `game_name` text, -- game name. the name of the game.
    primary key (id),
    foreign key (genre_id) references genre(id)
);

CREATE TABLE game_platform (
    `id` integer, -- the unique identifier of the record.
    `game_publisher_id` integer, -- game publisher id. the id of the game publisher.
    `platform_id` integer, -- platform id. the id of the platform.
    `release_year` integer, -- release year. the release year of the game.
    primary key (id),
    foreign key (game_publisher_id) references game_publisher(id),
    foreign key (platform_id) references platform(id)
);

CREATE TABLE platform (
    `id` integer, -- the unique identifier of the game platform.
    `platform_name` text, -- the name of the platform. The game platform, or the platform on which a game is played, can have a significant effect on the game. The platform can determine what hardware and software the game can run on, as well as the technical capabilities of the game. The platform can also affect the audience for the game, as different platforms may attract different groups of players.
    primary key (id)
);

CREATE TABLE region_sales (
    `region_id` integer, -- the id of the region.
    `game_platform_id` integer, -- game platform id. the id of the game platform.
    `num_sales` real, -- number sales. the number of sales in this region. The number of games sold in the region = num_sales * 100000. The game platform with higher num_sales is more popular in the region.
    foreign key (game_platform_id) references game_platform(id),
    foreign key (region_id) references region(id)
);

CREATE TABLE region (
    `id` integer, -- the unique identifier of the region.
    `region_name` text, -- the region name.
    primary key (id)
);

CREATE TABLE publisher (
    `id` integer, -- the unique identifier of the game publisher.
    `publisher_name` text, -- the name of the publisher. The publisher is the company or organization that finances, produces, and distributes the game. The publisher can influence the development of the game by providing resources, guidance, and support to the game's development team. The publisher can also affect the marketing and distribution of the game, as they are responsible for promoting the game and making it available to players.
    primary key (id)
);

CREATE TABLE game_publisher (
    `id` integer, -- the unique identifier of the game publisher.
    `game_id` integer, -- game id. the id of the game.
    `publisher_id` integer, -- publisher id. the id of the publisher. If game A and game B were published by the same publisher, the user who likes game A may also like game B if the user is a fan of the publisher company.
    primary key (id),
    foreign key (game_id) references game(id),
    foreign key (publisher_id) references publisher(id)
);

