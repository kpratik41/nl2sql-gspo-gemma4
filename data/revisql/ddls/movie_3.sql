CREATE TABLE film_text (
    `film_id` integer, -- film id. unique id number identifying the film.
    `title` text, -- title of the film.
    `description` text, -- main content of the film.
    primary key (film_id)
);

CREATE TABLE customer (
    `customer_id` integer, -- country id. unique id number identifying the country.
    `store_id` integer, -- store id. unique id number identifying the store.
    `first_name` text, -- first name. First name of the customer.
    `last_name` text, -- last name. Last name of the customer.
    `email` text, -- Email address of the customer.
    `address_id` integer , -- address id. Address id number of the customer.
    `active` integer, -- Wether the customer is active or not. 1: active. 0: not active.
    `create_date` datetime, -- create date. The date when the record is created.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (customer_id)
);

CREATE TABLE inventory (
    `inventory_id` integer , -- inventory id. unique id number identifying the inventory.
    `film_id` integer , -- film id. unique id number identifying the film.
    `store_id` integer, -- store id. id of the store.
    `last_update` datetime, -- last update. the last update time of the film.
    primary key (inventory_id)
);

CREATE TABLE rental (
    `rental_id` integer , -- rental id. unique id number identifying the rental.
    `rental_date` datetime, -- rental date. date when the rental occurs.
    `inventory_id` integer , -- inventory id. id number identifying the inventory.
    `customer_id` integer , -- customer id. id number identifying the customer.
    `return_date` datetime, -- return date. date when the rental returns.
    `staff_id` integer , -- staff id. id number identifying the staff.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (rental_id)
);

CREATE TABLE staff (
    `staff_id` integer , -- staff id. unique id number identifying the staff.
    `first_name` text, -- first name. First name of the actor.
    `last_name` text, -- last name. Last name of the actor. full name = (first name, last name).
    `address_id` integer, -- address id. id number identifying the address.
    `picture` blob, -- picture of the staff.
    `email` text, -- email of the staff.
    `store_id` integer , -- store id. id number identifying the store.
    `active` integer, -- Whether the staff is active or not. 1: active. 0: not active.
    `username` text, -- username to login the staff.
    `password` text, -- password to login the staff.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (staff_id)
);

CREATE TABLE film_category (
    `film_id` integer, -- film id. unique id number identifying the film.
    `category_id` integer, -- category id. id number identifying the category.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (film_id, category_id)
);

CREATE TABLE film (
    `film_id` integer , -- film id. unique id number identifying the film.
    `title` text, -- title of the film.
    `description` text, -- main content of the film.
    `release_year` text, -- release year. the year when the film is released.
    `language_id` integer , -- language id. the language id of the film.
    `original_language_id` integer , -- original language id. the original language id of the film.
    `rental_duration` integer, -- rental duration. how long this film can be rent at once. days. price / day = rental_rate / retal_duration.
    `rental_rate` real, -- rental rate. the rate of renting this film. higher -> expensive.
    `length` integer, -- Duration time of the film screening. minutes.
    `replacement_cost` real, -- replacement cost. cost of replacing this film.
    `rating` text, -- The Motion Picture Association film rating. G ?General Audiences. PG ? Parental Guidance Suggested. PG-13 ? Parents Strongly Cautioned. R ? Restricted. NC-17 ? Adults Only.
    `special_features` text, -- special features. features of this film.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (film_id)
);

CREATE TABLE film_actor (
    `actor_id` integer , -- actor id. unique id number identifying the actor.
    `film_id` integer , -- film id. id number identifying the film.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (actor_id, film_id)
);

CREATE TABLE payment (
    `payment_id` integer , -- payment id. unique id number identifying the payment.
    `customer_id` integer , -- customer id. unique id number identifying the customer.
    `staff_id` integer , -- staff id. unique id number identifying the staff.
    `rental_id` integer, -- rental id. unique id number identifying the rental.
    `amount` real, -- unique id number identifying the amount.
    `payment_date` datetime, -- payment date. the date when the payment ocurs.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (payment_id)
);

CREATE TABLE category (
    `category_id` integer , -- category id. unique id number identifying the category.
    `name` text, -- name of the category.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (category_id)
);

CREATE TABLE country (
    `country_id` integer , -- country id. unique id number identifying the country.
    `country` text, -- the name of the country. number identifying the country.  Africa: (Algeria, Angola, Cameroon, Chad, Congo, The Democratic Republic of the, Egypt, Ethiopia, Gambia...).  Asia: (Afghanistan, Armenia, Azerbaijan,. Bahrain, Bangladesh, Brunei, Cambodia, China, Hong Kong, India, Indonesia, Iran, Iraq, Israel...).  Oceania (American Samoa, Australia, French Polynesia...).  North America (Anguilla, Canada, Dominican Republic, Ecuador, Greenland...).  South America (Argentina, Bolivia, Brazil, Chile,. Colombia, Ecuador, French Guiana....).  Europe (Austria, Belarus, Bulgaria, Czech Republic, Estonia, Faroe Islands, Finland, France, Germany, Greece, Holy See (Vatican City State), Hungary, Italy...). details: https://worldpopulationreview.com/country-rankings/list-of-countries-by-continent. question can mention i.e., Europe instead of Austria, etc.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (country_id)
);

CREATE TABLE store (
    `store_id` integer , -- store id. unique id number identifying the store.
    `manager_staff_id` integer , -- manager staff id. id number identifying the manager staff.
    `address_id` integer , -- address id. id number identifying the address.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (store_id)
);

CREATE TABLE actor (
    `actor_id` integer , -- actor id. unique id number identifying the actor.
    `first_name` text, -- first name. First name of the actor.
    `last_name` text, -- last name. Last name of the actor.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (actor_id)
);

CREATE TABLE city (
    `city_id` integer , -- city id. unique id number identifying the city.
    `city` text, -- name of the city.
    `country_id` integer , -- country id. number identifying the country which the city belongs to.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (city_id)
);

CREATE TABLE address (
    `address_id` integer , -- address id. unique id number identifying the address.
    `address` text, -- The first address line.
    `address2` text, -- address 2. the second address line. address2 is the additional address if any.
    `district` text, -- No description.
    `city_id` integer unsigned, -- No description.
    `postal_code` text, -- postal code. a postal code is a series of letters or digits or both, sometimes including spaces or punctuation, included in a postal address for the purpose of sorting mail.
    `phone` text, -- phone number.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (address_id)
);

CREATE TABLE language (
    `language_id` integer , -- language id. unique id number identifying the language.
    `name` text, -- name of the language.
    `last_update` datetime, -- last update. The time of the latest update.
    primary key (language_id)
);

