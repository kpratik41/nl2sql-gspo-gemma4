CREATE TABLE app_all (
    `app_id` integer, -- No description.
    PRIMARY KEY (app_id)
);

CREATE TABLE sample_submission (
    `device_id` integer, -- No description.
    `F23-` real, -- No description.
    `F24-26` real, -- No description.
    `F27-28` real, -- No description.
    `F29-32` real, -- No description.
    `F33-42` real, -- No description.
    `F43+` real, -- No description.
    `M22-` real, -- No description.
    `M23-26` real, -- No description.
    `M27-28` real, -- No description.
    `M29-31` real, -- No description.
    `M32-38` real, -- No description.
    `M39+` real, -- No description.
    PRIMARY KEY (device_id)
);

CREATE TABLE phone_brand_device_model2 (
    `device_id` integer, -- device id. the id number of the device.
    `phone_brand` text, -- phone brand. phone_brand has duplicated values since some device models belong to the same brand.
    `device_model` text, -- device model. phone_brand has duplicated values since some device models belong to the same brand.
    PRIMARY KEY (device_id,phone_brand,device_model)
);

CREATE TABLE gender_age_test (
    `device_id` integer, -- No description.
    PRIMARY KEY (device_id)
);

CREATE TABLE app_events_relevant (
    `event_id` integer, -- No description.
    `app_id` integer, -- No description.
    `is_installed` integer, -- No description.
    `is_active` integer, -- No description.
    PRIMARY KEY (event_id,app_id),
    FOREIGN KEY (event_id) REFERENCES events_relevant (event_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (app_id) REFERENCES app_all (app_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE label_categories (
    `label_id` integer, -- label id. unique id of label.
    `category` text, -- category of the label.
    PRIMARY KEY (label_id)
);

CREATE TABLE gender_age (
    `device_id` integer, -- device id. unique number of devices.
    `gender` text, -- gender of the user who uses this device.
    `age` integer, -- age of the user who uses this device.  M: male;  F: female.
    `group` text, -- group of the ages.
    PRIMARY KEY (device_id),
    FOREIGN KEY (device_id) REFERENCES phone_brand_device_model2 (device_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE app_labels (
    `app_id` integer, -- app id. id of the app user.
    `label_id` integer, -- label id. id of labels represents which behavior category that each user belongs to.
    FOREIGN KEY (label_id) REFERENCES label_categories (label_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (app_id) REFERENCES app_all (app_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE gender_age_train (
    `device_id` integer, -- No description.
    `gender` text, -- No description.
    `age` integer, -- No description.
    `group` text, -- No description.
    PRIMARY KEY (device_id)
);

CREATE TABLE events (
    `event_id` integer, -- event id. unique id number referring to the event.
    `device_id` integer, -- device id. id number referring the device.
    `timestamp` datetime, -- the time of the event.
    `longitude` real, -- longitude. the location / coordinate = (longitude, latitude).
    `latitude` real, -- latitude.
    PRIMARY KEY (event_id)
);

CREATE TABLE events_relevant (
    `event_id` integer, -- No description.
    `device_id` integer, -- No description.
    `timestamp` datetime, -- No description.
    `longitude` real, -- No description.
    `latitude` real, -- No description.
    PRIMARY KEY (event_id),
    FOREIGN KEY (device_id) REFERENCES gender_age (device_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE app_events (
    `event_id` integer, -- event id. the id of events.
    `app_id` integer, -- app id. the id of app users. each app_id represents for an user.
    `is_installed` integer, -- is installed. whether this app is installed or not.  0: no.  1: yes: installed.
    `is_active` integer, -- is active. whether this user is active or not.
    PRIMARY KEY (event_id,app_id),
    FOREIGN KEY (event_id) REFERENCES events (event_id) ON DELETE CASCADE ON UPDATE CASCADE
);

