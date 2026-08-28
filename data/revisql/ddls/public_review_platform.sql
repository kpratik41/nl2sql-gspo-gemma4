CREATE TABLE Reviews (
    `business_id` integer, -- business id. the number identifying the business.
    `user_id` integer, -- user id. the number identifying the user who comments on this business.
    `review_stars` integer, -- review stars. review on this business. 5 – Great experience. 4 – Good experience. 3 – Average experience. 2 – Bad experience. 1 - Terrible experience.
    `review_votes_funny` text, -- review votes funny. the amount of funny votes that the user received for the review. If the reviews receive an “Uber” number of votes for funny, they will also receive an “Uber”, “High” or “Medium” number of votes for “useful” and “cool”.
    `review_votes_useful` text, -- review votes useful. how many useful votes that the user received for the review.
    `review_votes_cool` text, -- review votes cool. how many cool votes that the user received for the review.
    `review_length` text, -- review length. The length of the review written by the user.
    primary key (business_id, user_id)
);

CREATE TABLE Compliments (
    `compliment_id` integer, -- compliment id.
    `compliment_type` text, -- compliment type.
    primary key (compliment_id)
);

CREATE TABLE Business_Attributes (
    `attribute_id` integer, -- attribute id. id number identifying the attribute.
    `business_id` integer, -- business id. id number identifying the business.
    `attribute_value` text, -- attribute value. sort of the attributes for each business. “None”, “No” or “FALSE” means the business does not have the attribute.
    primary key (attribute_id, business_id)
);

CREATE TABLE Business_Categories (
    `business_id` integer, -- business id. id number identifying the business.
    `category_id` integer, -- category id. id number identifying the categories.
    primary key (business_id, category_id)
);

CREATE TABLE Attributes (
    `attribute_id` integer, -- attribute id. unique number identifying the attribute.
    `attribute_name` text, -- attribute name. the name of the attribute.
    primary key (attribute_id)
);

CREATE TABLE Users_Compliments (
    `compliment_id` integer, -- compliment id. the id number indicating the compliment.
    `user_id` integer, -- user id. the id number indicating the user.
    `number_of_compliments` text, -- number of compliments. how many compliments a user has received from other users. more number_of_compliments indicates this user is more welcome or he / she is high-quality user.
    primary key (compliment_id, user_id)
);

CREATE TABLE Business (
    `business_id` integer, -- business id. unique number identifying the business.
    `active` text, -- whether the business is still actively running until now. commonsense reasoning:. ï¿½ "True": the business is still running. ï¿½ "False": the business is closed or not running now.
    `city` text, -- The city where the business is located.
    `state` text, -- The state where the business is located.
    `stars` real, -- ratings of the business. 5 ï¿½ Great experience. 4 ï¿½ Good experience. 3 ï¿½ Average experience. 2 ï¿½ Bad experience. 1 - Terrible experience. ï¿½ the rating of >3 stars referring to "wonderful experience" or positive comments and vice versa.
    `review_count` text, -- review count. the total number of reviews the users have written for a business. ï¿½ If a business has a low total review count and a high star rating of >3, it means there is a low veracity of reviews. ï¿½ higher review count and with high star rating of > 3 means this business is more popular or more appealing to users.
    primary key (business_id)
);

CREATE TABLE Users (
    `user_id` integer, -- user id. the unique id number identifying which user.
    `user_yelping_since_year` integer, -- user yelping since year. the time when the user join Yelp.
    `user_average_stars` real, -- user average stars. the average ratings of all review.
    `user_votes_funny` text, -- user votes funny. total number of funny votes sent by the user.
    `user_votes_useful` text, -- user votes useful. how many useful votes created by the user.
    `user_votes_cool` text, -- user votes cool. how many cool votes created by the user.
    `user_review_count` text, -- user review count. total number of reviews the user has written.
    `user_fans` text, -- user fans. total number of fans / followers the user has. Users with “Uber” number of fans indicate that they have sent an “Uber” number of ‘cool’, ‘useful’ and ‘funny” votes.
    primary key (user_id)
);

CREATE TABLE Tips (
    `business_id` integer, -- business id. the number identifying the business.
    `user_id` integer, -- user id. the number identifying the user who comments on this business.
    `likes` integer, -- Likes. how many likes of this tips. more likes mean this tip is more valuable.
    `tip_length` text, -- tip length. length of the tip.
    primary key (business_id, user_id)
);

CREATE TABLE Elite (
    `user_id` integer, -- user id. id number identifying the users.
    `year_id` integer, -- year id. id number identifying the year.
    primary key (user_id, year_id)
);

CREATE TABLE Checkins (
    `business_id` integer, -- business id. id number identifying the business.
    `day_id` integer, -- day id. id number identifying each day of the week.
    `label_time_0` text, -- indicates times of checkins on a business. label_time_0: 12:00 a.m. label_time_23: 23:00 p.m. If the label_time recorded "None" for check-in on one day, then it means the business is closed on that day.
    `label_time_1` text, -- No description.
    `label_time_2` text, -- No description.
    `label_time_3` text, -- No description.
    `label_time_4` text, -- No description.
    `label_time_5` text, -- No description.
    `label_time_6` text, -- No description.
    `label_time_7` text, -- No description.
    `label_time_8` text, -- No description.
    `label_time_9` text, -- No description.
    `label_time_10` text, -- No description.
    `label_time_11` text, -- No description.
    `label_time_12` text, -- No description.
    `label_time_13` text, -- No description.
    `label_time_14` text, -- No description.
    `label_time_15` text, -- No description.
    `label_time_16` text, -- No description.
    `label_time_17` text, -- No description.
    `label_time_18` text, -- No description.
    `label_time_19` text, -- No description.
    `label_time_20` text, -- No description.
    `label_time_21` text, -- No description.
    `label_time_22` text, -- No description.
    `label_time_23` text, -- No description.
    primary key (business_id, day_id)
);

CREATE TABLE Days (
    `day_id` integer, -- day id. the unique id identifying the day of the week.
    `day_of_week` text, -- day of week. indicate the day of the week.
    primary key (day_id)
);

CREATE TABLE Categories (
    `category_id` integer, -- category id.
    `category_name` text, -- category name.
    primary key (category_id)
);

CREATE TABLE Years (
    `year_id` integer, -- year id. the unique number identifying the year.
    `actual_year` integer, -- actual year.
    primary key (year_id)
);

CREATE TABLE Business_Hours (
    `business_id` integer, -- business id. id number identifying the business.
    `day_id` integer, -- day id. id number identifying each day of the week.
    `opening_time` text, -- opening time. opening time of the business.
    `closing_time` text, -- closing time. closing time of the business. how much time does this business open: closing_time - opening_time.
    primary key (business_id, day_id)
);

