CREATE TABLE Dish (
    `id` integer, -- unique id number indicating the dishes.
    `name` text, -- the name of the dish.
    `description` text, -- description of the dish. (no value).
    `menus_appeared` integer, -- menus appeared. how many menus have this dish.
    `times_appeared` integer, -- times appeared. how many times this dish appears in the menu. if times_appeared > menus_appeard: this dish appeared in a menu more than once.
    `first_appeared` integer, -- first appeared. the year that this dish appears first. 1. the year outside of [1851, 2012], it means this data is not correct. 2. if this dish lasts longer (last_appeared - first_appeard), it means its history is long or historical / classical.
    `last_appeared` integer, -- last appeared. the year that this dish appears the last time. 1. the year outside of [1851, 2012], it means this data is not correct. 2. if this dish lasts longer (last_appeared - first_appeard), it means its history is long or historical / classical.
    `lowest_price` real, -- lowest price. the lowest price of the dish. 0: for free.
    `highest_price` real, -- highest price. the highest price of the dish.
    primary key (id)
);

CREATE TABLE Menu (
    `id` integer, -- the unique number identifying the menu.
    `name` text, -- the name of the menu. if the value is not null or empty, it means this menu has special dishes. otherwise, this menu is general and nothing special.
    `sponsor` text, -- the sponsor of this menu. if the value is null or empyt, it means this meanu is DIY by the restaurant.
    `event` text, -- the event that the menu was created for.
    `venue` text, -- the venue that the menu was created for.
    `place` text, -- the location that the menu was used.
    `physical_description` text, -- physical description. physical description of the menu.
    `occasion` text, -- occasion of the menu.
    `notes` text, -- notes.
    `call_number` text, -- call number. if null: not support for taking out or booking in advance.
    `keywords` text, -- keywords. not useful.
    `language` text, -- language. not useful.
    `date` date, -- the date that this menu was created.
    `location` text, -- the location that the menu was used.
    `location_type` text, -- not useful.
    `currency` text, -- the currency that the menu was used.
    `currency_symbol` text, -- the currency symbol.
    `status` text, -- status of the menu.
    `page_count` integer, -- page count. the number of pages of this menu.
    `dish_count` integer, -- dish count. the number of dishes of this menu.
    primary key (id)
);

CREATE TABLE MenuPage (
    `id` integer, -- the unique id number indentifying the menupage.
    `menu_id` integer, -- menu id. the id of the menu.
    `page_number` integer, -- page number. the page number.
    `image_id` real, -- image id. the id of the image.
    `full_height` integer, -- full height. full height of the menu page. (mm).
    `full_width` integer, -- full width. full width of the menu page. (mm).
    `uuid` text, -- No description.
    primary key (id),
    foreign key (menu_id) references Menu(id)
);

CREATE TABLE MenuItem (
    `id` integer, -- the unique id representing the menu item.
    `menu_page_id` integer, -- the id of menu page.
    `price` real, -- the price of this dish (menu item).
    `high_price` real, -- high price. high price of this dish.
    `dish_id` integer, -- the id of the dish.
    `created_at` text, -- created at. the dates when the item was created.
    `updated_at` text, -- updated at. the dates when the item was updated.
    `xpos` real, -- x position. x-axis position of the dish in this menu page.
    `ypos` real, -- y position. y-axis position of the dish in this menu page.
    primary key (id),
    foreign key (dish_id) references Dish(id),
    foreign key (menu_page_id) references MenuPage(id)
);

