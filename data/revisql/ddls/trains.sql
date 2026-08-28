CREATE TABLE trains (
    `id` integer, -- the unique id representing the trains.
    `direction` text, -- the direction of trains that are running. â¢ east; â¢ west;
    PRIMARY KEY (id)
);

CREATE TABLE cars (
    `id` integer, -- the unique id number representing the cars.
    `train_id` integer, -- train id. the counterpart id for trains that the cars belong to.
    `position` integer, -- postion id of cars in the trains. 1-4:. 1: head car. 4: tail car.
    `shape` text, -- shape of the cars. â¢ rectangle. â¢ bucket. â¢ u_shaped. â¢ hexagon. â¢ elipse. regular shape:. rectangle, u_shaped, hexagon.
    `len` text, -- length. length of the cars. â¢ short. â¢ long.
    `sides` text, -- sides of the cars. â¢ not_double. â¢ double.
    `roof` text, -- roof of the cars. â¢ none: the roof is open. â¢ peaked. â¢ flat. â¢ arc. â¢ jagged.
    `wheels` integer, -- wheels of the cars. â¢ 2:. â¢ 3:.
    `load_shape` text, -- load shape. â¢ circle. â¢ hexagon. â¢ triangle. â¢ rectangle. â¢ diamond.
    `load_num` integer, -- load number. 0-3:. â¢ 0: empty load. â¢ 3: full load.
    PRIMARY KEY (id),
    FOREIGN KEY (train_id) REFERENCES trains (id) ON DELETE CASCADE ON UPDATE CASCADE
);

