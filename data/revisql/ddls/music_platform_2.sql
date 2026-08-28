CREATE TABLE categories (
    `podcast_id` text, -- podcast id. The unique id of the podcast.
    `category` text, -- category of the podcast. This usually represents the topic of the podcast.
    primary key (podcast_id, category)
);

CREATE TABLE podcasts (
    `podcast_id` text, -- podcast id. The unique id of the podcast.
    `itunes_id` integer, -- itunes id. The unique id of the itunes.
    `slug` text, -- The slug of the podcast. It usually has the same name with the "title" column.
    `itunes_url` text, -- itunes url. The iTunes url of the podcast. Can visit the webpage of the podcast by clicking the itunes_url.
    `title` text, -- The title of the podcast. It usually has a very similar name to the "slug" column.
    primary key (podcast_id)
);

CREATE TABLE reviews (
    `podcast_id` text, -- podcast id. The unique id of the podcast.
    `title` text, -- The title of the podcast review. This usually is the abstract of the whole review.
    `content` text, -- The content of the podcast review. This usually is the detailed explanation of the podcast review title.
    `rating` integer, -- The rating of the podcast review. Allowed values: 0 to 5. This rating is highly related with "title" and "content".
    `author_id` text, -- author id. The author id of the podcast review.
    `created_at` text, -- created at. The date and time of the podcast review creation. In the format of "Date time". e.g., 2018-05-09T18:14:32-07:00.

);

CREATE TABLE runs (
    `run_at` text, -- run at. The date and time of the podcast review creation.
    `max_rowid` integer, -- max row id. The id of max row of this run.
    `reviews_added` integer, -- reviews added. The number of reviews added in this run.

);

