CREATE TABLE actor (
  actor_id numeric NOT NULL , -- example: [1, 2, 3]
  first_name VARCHAR(45) NOT NULL, -- example: ['PENELOPE', 'NICK', 'ED']
  last_name VARCHAR(45) NOT NULL, -- example: ['AKROYD', 'ALLEN', 'ASTAIRE']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:51:59', '2021-03-06 15:52:00']
  PRIMARY KEY  (actor_id)
  )

CREATE TABLE country (
  country_id SMALLINT NOT NULL, -- example: [1, 2, 3]
  country VARCHAR(50) NOT NULL, -- example: ['Afghanistan', 'Algeria', 'American Samoa']
  last_update TIMESTAMP, -- example: ['2021-03-06 15:51:49']
  PRIMARY KEY  (country_id)
)

CREATE TABLE city (
  city_id int NOT NULL, -- example: [1, 2, 3]
  city VARCHAR(50) NOT NULL, -- example: ['A Corua (La Corua)', 'Abha', 'Abu Dhabi']
  country_id SMALLINT NOT NULL, -- example: [1, 2, 3]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:51:49', '2021-03-06 15:51:50', '2021-03-06 15:51:51']
  PRIMARY KEY  (city_id),
  CONSTRAINT fk_city_country FOREIGN KEY (country_id) REFERENCES country (country_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE address (
  address_id int NOT NULL, -- example: [1, 2, 3]
  address VARCHAR(50) NOT NULL, -- example: ['47 MySakila Drive', '28 MySQL Boulevard', '23 Workhaven Lane']
  address2 VARCHAR(50) DEFAULT NULL, -- example: NULL
  district VARCHAR(20) NOT NULL, -- example: [' ']
  city_id INT  NOT NULL, -- example: [1, 2, 3]
  postal_code VARCHAR(10) DEFAULT NULL, -- example: ['35200', '17886', '83579']
  phone VARCHAR(20) NOT NULL, -- example: [' ']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:51:54', '2021-03-06 15:51:55', '2021-03-06 15:51:56']
  PRIMARY KEY  (address_id),
  CONSTRAINT fk_address_city FOREIGN KEY (city_id) REFERENCES city (city_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE language (
  language_id SMALLINT NOT NULL , -- example: [1, 2, 3]
  name CHAR(20) NOT NULL, -- example: ['English', 'Italian', 'Japanese']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:51:48']
  PRIMARY KEY (language_id)
)

CREATE TABLE category (
  category_id SMALLINT NOT NULL, -- example: [1, 2, 3]
  name VARCHAR(25) NOT NULL, -- example: ['Action', 'Animation', 'Children']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:00']
  PRIMARY KEY  (category_id)
)

CREATE TABLE customer (
  customer_id INT NOT NULL, -- example: [1, 2, 3]
  store_id INT NOT NULL, -- example: [1, 2]
  first_name VARCHAR(45) NOT NULL, -- example: ['MARY', 'PATRICIA', 'LINDA']
  last_name VARCHAR(45) NOT NULL, -- example: ['ABNEY', 'ADAM', 'ADAMS']
  email VARCHAR(50) DEFAULT NULL, -- example: ['MARY.SMITH@sakilacustomer.org', 'PATRICIA.JOHNSON@sakilacustomer.org', 'LINDA.WILLIAMS@sakilacustomer.org']
  address_id INT NOT NULL, -- example: [5, 6, 7]
  active CHAR(1) DEFAULT 'Y' NOT NULL, -- example: ['1', '0']
  create_date TIMESTAMP NOT NULL, -- example: ['2006-02-14 22:04:36.000', '2006-02-14 22:04:37.000']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:53:36', '2021-03-06 15:53:37', '2021-03-06 15:53:38']
  PRIMARY KEY  (customer_id),
  CONSTRAINT fk_customer_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE,
  CONSTRAINT fk_customer_address FOREIGN KEY (address_id) REFERENCES address (address_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE film (
  film_id int NOT NULL, -- example: [1, 2, 3]
  title VARCHAR(255) NOT NULL, -- example: ['ACADEMY DINOSAUR', 'ACE GOLDFINGER', 'ADAPTATION HOLES']
  description BLOB SUB_TYPE TEXT DEFAULT NULL, -- example: ['A Epic Drama of a Feminist And a Mad Scientist who must Battle a Teacher in The Canadian Rockies', 'A Astounding Epistle of a Database Administrator And a Explorer who must Find a Car in Ancient China', 'A Astounding Reflection of a Lumberjack And a Car who must Sink a Lumberjack in A Baloon Factory']
  release_year VARCHAR(4) DEFAULT NULL, -- example: ['2006']
  language_id SMALLINT NOT NULL, -- example: [1]
  original_language_id SMALLINT DEFAULT NULL, -- example: NULL
  rental_duration SMALLINT  DEFAULT 3 NOT NULL, -- example: [6, 3, 7]
  rental_rate DECIMAL(4,2) DEFAULT 4.99 NOT NULL, -- example: [0.99, 4.99, 2.99]
  length SMALLINT DEFAULT NULL, -- example: [86, 48, 50]
  replacement_cost DECIMAL(5,2) DEFAULT 19.99 NOT NULL, -- example: [20.99, 12.99, 18.99]
  rating VARCHAR(10) DEFAULT 'G', -- example: ['PG', 'G', 'NC-17']
  special_features VARCHAR(100) DEFAULT NULL, -- example: ['Deleted Scenes,Behind the Scenes', 'Trailers,Deleted Scenes', 'Commentaries,Behind the Scenes']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:00', '2021-03-06 15:52:01', '2021-03-06 15:52:02']
  PRIMARY KEY  (film_id),
  CONSTRAINT CHECK_special_features CHECK(special_features is null or
                                                           special_features like '%Trailers%' or -- example: ['Deleted Scenes,Behind the Scenes', 'Trailers,Deleted Scenes', 'Commentaries,Behind the Scenes']
                                                           special_features like '%Commentaries%' or -- example: ['Deleted Scenes,Behind the Scenes', 'Trailers,Deleted Scenes', 'Commentaries,Behind the Scenes']
                                                           special_features like '%Deleted Scenes%' or -- example: ['Deleted Scenes,Behind the Scenes', 'Trailers,Deleted Scenes', 'Commentaries,Behind the Scenes']
                                                           special_features like '%Behind the Scenes%'), -- example: ['Deleted Scenes,Behind the Scenes', 'Trailers,Deleted Scenes', 'Commentaries,Behind the Scenes']
  CONSTRAINT CHECK_special_rating CHECK(rating in ('G','PG','PG-13','R','NC-17')),
  CONSTRAINT fk_film_language FOREIGN KEY (language_id) REFERENCES language (language_id) ,
  CONSTRAINT fk_film_language_original FOREIGN KEY (original_language_id) REFERENCES language (language_id)
)

CREATE TABLE film_actor (
  actor_id INT NOT NULL, -- example: [1, 2, 3]
  film_id  INT NOT NULL, -- example: [1, 2, 3]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:45', '2021-03-06 15:52:46', '2021-03-06 15:52:47']
  PRIMARY KEY  (actor_id,film_id),
  CONSTRAINT fk_film_actor_actor FOREIGN KEY (actor_id) REFERENCES actor (actor_id) ON DELETE NO ACTION ON UPDATE CASCADE,
  CONSTRAINT fk_film_actor_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE film_category (
  film_id INT NOT NULL, -- example: [1, 2, 3]
  category_id SMALLINT  NOT NULL, -- example: [1, 2, 3]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:53:28', '2021-03-06 15:53:29', '2021-03-06 15:53:30']
  PRIMARY KEY (film_id, category_id),
  CONSTRAINT fk_film_category_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE,
  CONSTRAINT fk_film_category_category FOREIGN KEY (category_id) REFERENCES category (category_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE film_text (
  film_id SMALLINT NOT NULL, -- example: NULL
  title VARCHAR(255) NOT NULL, -- example: NULL
  description BLOB SUB_TYPE TEXT, -- example: NULL
  PRIMARY KEY  (film_id)
)

CREATE TABLE inventory (
  inventory_id INT NOT NULL, -- example: [1, 2, 3]
  film_id INT NOT NULL, -- example: [1, 2, 3]
  store_id INT NOT NULL, -- example: [1, 2]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:08', '2021-03-06 15:52:09', '2021-03-06 15:52:10']
  PRIMARY KEY  (inventory_id),
  CONSTRAINT fk_inventory_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE,
  CONSTRAINT fk_inventory_film FOREIGN KEY (film_id) REFERENCES film (film_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE staff (
  staff_id SMALLINT NOT NULL, -- example: [1, 2]
  first_name VARCHAR(45) NOT NULL, -- example: ['Mike', 'Jon']
  last_name VARCHAR(45) NOT NULL, -- example: ['Hillyer', 'Stephens']
  address_id INT NOT NULL, -- example: [3, 4]
  picture BLOB DEFAULT NULL, -- example: NULL
  email VARCHAR(50) DEFAULT NULL, -- example: ['Mike.Hillyer@sakilastaff.com', 'Jon.Stephens@sakilastaff.com']
  store_id INT NOT NULL, -- example: [1, 2]
  active SMALLINT DEFAULT 1 NOT NULL, -- example: [1]
  username VARCHAR(16) NOT NULL, -- example: ['Mike', 'Jon']
  password VARCHAR(40) DEFAULT NULL, -- example: ['8cb2237d0679ca88db6464eac60da96345513964']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:00']
  PRIMARY KEY  (staff_id),
  CONSTRAINT fk_staff_store FOREIGN KEY (store_id) REFERENCES store (store_id) ON DELETE NO ACTION ON UPDATE CASCADE,
  CONSTRAINT fk_staff_address FOREIGN KEY (address_id) REFERENCES address (address_id) ON DELETE NO ACTION ON UPDATE CASCADE
)

CREATE TABLE store (
  store_id INT NOT NULL, -- example: [1, 2]
  manager_staff_id SMALLINT NOT NULL, -- example: [1, 2]
  address_id INT NOT NULL, -- example: [1, 2]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:52:00']
  PRIMARY KEY  (store_id),
  CONSTRAINT fk_store_staff FOREIGN KEY (manager_staff_id) REFERENCES staff (staff_id) ,
  CONSTRAINT fk_store_address FOREIGN KEY (address_id) REFERENCES address (address_id)
)

CREATE TABLE payment (
  payment_id int NOT NULL, -- example: [1, 2, 3]
  customer_id INT  NOT NULL, -- example: [1, 2, 3]
  staff_id SMALLINT NOT NULL, -- example: [1, 2]
  rental_id INT DEFAULT NULL, -- example: [76, 573, 1185]
  amount DECIMAL(5,2) NOT NULL, -- example: [2.99, 0.99, 5.99]
  payment_date TIMESTAMP NOT NULL, -- example: ['2005-05-25 11:30:37.000', '2005-05-28 10:35:23.000', '2005-06-15 00:54:12.000']
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:55:57', '2021-03-06 15:55:58', '2021-03-06 15:55:59']
  PRIMARY KEY  (payment_id),
  CONSTRAINT fk_payment_rental FOREIGN KEY (rental_id) REFERENCES rental (rental_id) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT fk_payment_customer FOREIGN KEY (customer_id) REFERENCES customer (customer_id) ,
  CONSTRAINT fk_payment_staff FOREIGN KEY (staff_id) REFERENCES staff (staff_id)
)

CREATE TABLE rental (
  rental_id INT NOT NULL, -- example: [1, 2, 3]
  rental_date TIMESTAMP NOT NULL, -- example: ['2005-05-24 22:53:30.000', '2005-05-24 22:54:33.000', '2005-05-24 23:03:39.000']
  inventory_id INT  NOT NULL, -- example: [1, 2, 3]
  customer_id INT  NOT NULL, -- example: [1, 2, 3]
  return_date TIMESTAMP DEFAULT NULL, -- example: ['2005-05-26 22:04:30.000', '2005-05-28 19:40:33.000', '2005-06-01 22:12:39.000']
  staff_id SMALLINT  NOT NULL, -- example: [1, 2]
  last_update TIMESTAMP NOT NULL, -- example: ['2021-03-06 15:53:41', '2021-03-06 15:53:42', '2021-03-06 15:53:43']
  PRIMARY KEY (rental_id),
  CONSTRAINT fk_rental_staff FOREIGN KEY (staff_id) REFERENCES staff (staff_id) ,
  CONSTRAINT fk_rental_inventory FOREIGN KEY (inventory_id) REFERENCES inventory (inventory_id) ,
  CONSTRAINT fk_rental_customer FOREIGN KEY (customer_id) REFERENCES customer (customer_id)
)

