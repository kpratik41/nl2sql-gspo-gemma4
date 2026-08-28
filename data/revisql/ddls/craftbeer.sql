CREATE TABLE breweries (
    `id` integer, -- unique ID of the breweries.
    `name` text, -- name of the breweries.
    `city` text, -- city.
    `state` text, -- state.
    primary key (id)
);

CREATE TABLE beers (
    `id` integer, -- unique id number of beers.
    `brewery_id` integer , -- brewery id. id number of the breweries.
    `abv` text, -- alcohol by volume. Alcohol by VolumeABV is the most common measurement of alcohol content in beer; it simply indicates how much of the total volume of liquid in a beer is made up of alcohol.
    `ibu` real, -- International Bitterness Units. IBU stands for International Bitterness Units, a scale to gauge the level of a beer's bitterness. More specifically, IBUs measure the parts per million of is humulone from hops in a beer, which gives beer bitterness.
    `name` text, -- name of beers.
    `style` text, -- style / sorts of beers.
    `ounces` real, -- ounces.
    primary key (id)
);

