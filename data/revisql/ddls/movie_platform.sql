CREATE TABLE movies (
    `movie_id` integer, -- ID related to the movie on Mubi.
    `movie_title` text, -- Name of the movie.
    `movie_release_year` integer, -- Release year of the movie.
    `movie_url` text, -- URL to the movie page on Mubi.
    `movie_title_language` text, -- By default, the title is in English. Only contains one value which is 'en'.
    `movie_popularity` integer, -- Number of Mubi users who love this movie.
    `movie_image_url` text, -- Image URL to the movie on Mubi.
    `director_id` text, -- ID related to the movie director on Mubi.
    `director_name` text, -- Full Name of the movie director.
    `director_url` text, -- URL to the movie director page on Mubi.
    primary key (movie_id)
);

CREATE TABLE ratings (
    `movie_id` integer, -- Movie ID related to the rating.
    `rating_id` integer, -- Rating ID on Mubi.
    `rating_url` text, -- URL to the rating on Mubi.
    `rating_score` integer, -- Rating score ranging from 1 (lowest) to 5 (highest). The score is proportional to the user's liking. The higher the score is, the more the user likes the movie.
    `rating_timestamp_utc` text, -- Timestamp for the movie rating made by the user on Mubi.
    `critic` text, -- Critic made by the user rating the movie. If value = "None", the user did not write a critic when rating the movie.
    `critic_likes` integer, -- Number of likes related to the critic made by the user rating the movie.
    `critic_comments` integer, -- Number of comments related to the critic made by the user rating the movie.
    `user_id` integer, -- ID related to the user rating the movie.
    `user_trialist` integer, -- whether user was a tralist when he rated the movie. 1 = the user was a trialist when he rated the movie. 0 = the user was not a trialist when he rated the movie.
    `user_subscriber` integer, -- No description.
    `user_eligible_for_trial` integer, -- No description.
    `user_has_payment_method` integer, -- No description.
    foreign key (movie_id) references movies(movie_id),
    foreign key (user_id) references lists_users(user_id),
    foreign key (rating_id) references ratings(rating_id),
    foreign key (user_id) references ratings_users(user_id)
);

CREATE TABLE ratings_users (
    `user_id` integer, -- ID related to the user rating the movie.
    `rating_date_utc` text, -- Rating date for the movie rating. YYYY-MM-DD.
    `user_trialist` integer, -- whether the user was a trialist when he rated the movie. 1 = the user was a trialist when he rated the movie. 0 = the user was not a trialist when he rated the movie.
    `user_subscriber` integer, -- whether the user was a subscriber when he rated the movie. 1 = the user was a subscriber when he rated the movie. 0 = the user was not a subscriber when he rated the movie.
    `user_avatar_image_url` text, -- URL to the user profile image on Mubi.
    `user_cover_image_url` text, -- URL to the user profile cover image on Mubi.
    `user_eligible_for_trial` integer, -- whether the user was eligible for trial when he rated the movie. 1 = the user was eligible for trial when he rated the movie. 0 = the user was not eligible for trial when he rated the movie.
    `user_has_payment_method` integer, -- whether the user was a paying subscriber when he rated the movie. 1 = the user was a paying subscriber when he rated the movie. 0 = the user was not a paying subscriber when he rated.

);

CREATE TABLE lists_users (
    `user_id` integer, -- ID related to the user who created the list.
    `list_id` integer, -- ID of the list on Mubi.
    `list_update_date_utc` text, -- Last update date for the list. YYYY-MM-DD.
    `list_creation_date_utc` text, -- Creation date for the list. YYYY-MM-DD.
    `user_trialist` integer, -- whether the user was a tralist when he created the list. 1 = the user was a trialist when he created the list. 0 = the user was not a trialist when he created the list.
    `user_subscriber` integer, -- whether the user was a subscriber when he created the list. 1 = the user was a subscriber when he created the list. 0 = the user was not a subscriber when he created the list.
    `user_avatar_image_url` text, -- User profile image URL on Mubi.
    `user_cover_image_url` text, -- User profile cover image URL on Mubi.
    `user_eligible_for_trial` text, -- whether the user was eligible for trial when he created the list. 1 = the user was eligible for trial when he created the list. 0 = the user was not eligible for trial when he created the list.
    `user_has_payment_method` text, -- whether the user was a paying subscriber when he created the list. 1 = the user was a paying subscriber when he created the list. 0 = the user was not a paying subscriber when he created the list.
    primary key (user_id, list_id),
    foreign key (list_id) references lists(list_id),
    foreign key (user_id) references lists(user_id)
);

CREATE TABLE lists (
    `user_id` integer, -- ID related to the user who created the list.
    `list_id` integer, -- ID of the list on Mubi.
    `list_title` text, -- Name of the list.
    `list_movie_number` integer, -- Number of movies added to the list.
    `list_update_timestamp_utc` text, -- Last update timestamp for the list.
    `list_creation_timestamp_utc` text, -- Creation timestamp for the list.
    `list_followers` integer, -- Number of followers on the list.
    `list_url` text, -- URL to the list page on Mubi.
    `list_comments` integer, -- Number of comments on the list.
    `list_description` text, -- List description made by the user.
    `list_cover_image_url` text, -- No description.
    `list_first_image_url` text, -- No description.
    `list_second_image_url` text, -- No description.
    `list_third_image_url` text, -- No description.
    primary key (list_id)
);

