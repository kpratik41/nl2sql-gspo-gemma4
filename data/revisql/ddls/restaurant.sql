CREATE TABLE geographic (
    `city` text, -- the city.
    `county` text, -- country. the country the city belongs to.
    `region` text, -- corresponding regions.
    primary key (city)
);

CREATE TABLE location (
    `id_restaurant` integer, -- id restaurant. the unique id for the restaurant.
    `street_num` integer, -- street number. the street number of the restaurant.
    `street_name` text, -- street name. the street name of the restaurant.
    `city` text, -- the city where the restaurant is located in.
    primary key (id_restaurant),
    foreign key (city) references geographic (city)
    foreign key (id_restaurant) references generalinfo (id_restaurant)
);

CREATE TABLE generalinfo (
    `id_restaurant` integer, -- id restaurant. the unique id for the restaurant.
    `label` text, -- the label of the restaurant.
    `food_type` text, -- food type. the food type.
    `city` text, -- the city where the restaurant is located in.
    `review` real, -- the review of the restaurant. the review rating is from 0.0 to 5.0. The high review rating is positively correlated with the overall level of the restaurant. The restaurant with higher review rating is usually more popular among diners.
    primary key (id_restaurant),
    foreign key (city) references geographic(city)
);

