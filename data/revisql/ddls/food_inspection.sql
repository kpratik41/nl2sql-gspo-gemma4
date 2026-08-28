CREATE TABLE inspections (
    `business_id` integer, -- business id. the unique id of the business.
    `score` integer, -- the inspection score. The scores range from 1 to 100, where 100 means that the establishment meets all required standards.
    `date` date, -- the inspection date.
    `type` text, -- the inspection type.
    FOREIGN KEY (business_id) REFERENCES businesses (business_id)
);

CREATE TABLE violations (
    `business_id` integer, -- business id. the unique id of the business.
    `date` date, -- the date of the violation.
    `violation_type_id` text, -- violation type id. the unique type id of the violation.
    `risk_category` text, -- risk category. risk category. High / Moderate / Low risk. High risks have more safety and health hazards than low risks.
    `description` text, -- the description of the violation.
    FOREIGN KEY (business_id) REFERENCES businesses (business_id)
);

CREATE TABLE businesses (
    `business_id` integer, -- business id. the unique id of the business.
    `name` text, -- the name of the eatery.
    `address` text, -- the eatery address.
    `city` text, -- the city where the eatery is located in.
    `postal_code` text, -- postal code. the postal code of the eatery.
    `latitude` real, -- the latitude of the position.
    `longitude` real, -- the longitude of the position. the distance between eatery A and eatery B = \sqrt{(latitude_A-latitude_B)^2 + (longitude_A-longitude_B)^2}.
    `phone_number` integer, -- phone number. the phone number of the eatery.
    `tax_code` text, -- tax code. the tax code of the eatery.
    `business_certificate` integer, -- business certificate. the business certificate number.
    `application_date` date, -- application date. the application date of the eatery.
    `owner_name` text, -- owner name. the owner's name.
    `owner_address` text, -- owner address. the owner's address.
    `owner_city` text, -- owner city. the city where the owner is located.
    `owner_state` text, -- owner state. the state where the owner is located.
    `owner_zip` text, -- owner zip. the zip code of the owner.
    PRIMARY KEY (business_id)
);

