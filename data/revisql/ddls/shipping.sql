CREATE TABLE driver (
    `driver_id` integer, -- driver id. Unique identifier for the driver.
    `first_name` text, -- first name. First given name of the driver.
    `last_name` text, -- last name. Family name of the driver. full name = first_name + last_name.
    `address` text, -- Street address of the driver's home.
    `city` text, -- City the driver lives in.
    `state` text, -- State the driver lives in. please mention its full name in the question, by referring to. https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971. e.g., NY --> New York.
    `zip_code` integer, -- zip code. postal code of the driver's address.
    `phone` text, -- telephone number of the driver.
    primary key (driver_id)
);

CREATE TABLE customer (
    `cust_id` integer, -- customer id. Unique identifier for the customer.
    `cust_name` text, -- customer name. Business name of the customer.
    `annual_revenue` integer, -- annual revenue. Annual revenue of the customer.
    `cust_type` text, -- customer type. Whether the customer is a manufacturer or a wholes.
    `address` text, -- Physical street address of the customer.
    `city` text, -- City of the customer's address.
    `state` text, -- State of the customer's address. please mention its full name in the question, by referring to. https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971. e.g., NY --> New York.
    `zip` real, -- Postal code of the customer's address.
    `phone` text, -- Telephone number to reach the customer.
    primary key (cust_id)
);

CREATE TABLE shipment (
    `ship_id` integer, -- ship id. Unique identifier of the shipment.
    `cust_id` integer, -- customer id. A reference to the customer table that indicates which customer the shipment is for.
    `weight` real, -- The number of pounds being transported on the shipment.
    `truck_id` integer, -- truck id. A reference to the truck table that indicates which truck is used in the shipment.
    `driver_id` integer, -- driver id. A reference to the driver table that indicates which driver transported the goods in the shipment.
    `city_id` integer, -- city id. A reference to the city table that indicates the destination of the shipment.
    `ship_date` text, -- ship date. the date the items were received by the driver. yyyy-mm-dd.
    primary key (ship_id),
    foreign key (cust_id) references customer(cust_id),
    foreign key (city_id) references city(city_id),
    foreign key (driver_id) references driver(driver_id),
    foreign key (truck_id) references truck(truck_id)
);

CREATE TABLE city (
    `city_id` integer, -- city id. unique identifier for the city.
    `city_name` text, -- city name. name of the city.
    `state` text, -- state in which the city is.
    `population` integer, -- population of the city.
    `area` real, -- square miles the city covers. population density (land area per capita) = area / population.
    primary key (city_id)
);

CREATE TABLE truck (
    `truck_id` integer, -- truck id. Unique identifier of the truck table.
    `make` text, -- The brand of the truck.  Peterbilt headquarter: Texas (TX).  Mack headquarter: North Carolina (NC).  Kenworth headquarter: Washington (WA). can ask question about headquarters of the truck.
    `model_year` integer, -- model year. The year the truck was manufactured. The truck with earlier model year means this truck is newer.
    primary key (truck_id)
);

