CREATE TABLE twitter (
    `TweetID` text, -- tweet id. the unique id of the tweet.
    `Weekday` text, -- the weekday that the tweet has been posted.
    `Hour` integer, -- the hour that the tweet has been posted.
    `Day` integer, -- the day that the tweet has been posted.
    `Lang` text, -- language. the language of the tweet.
    `IsReshare` text, -- is reshare. whether the tweet is reshared. A reshared tweet is when someone reshares or forwards a post to their own Twitter.  true: the tweet is reshared.  false: the tweet isn't reshared.
    `Reach` integer, -- the reach of the tweet. reach counts the number of unique users who have seen your tweet.
    `RetweetCount` integer, -- retweet count. the retweet count.
    `Likes` integer, -- the number of likes.
    `Klout` integer, -- the klout of the tweet.
    `Sentiment` real, -- the sentiment of the tweet.  >0: the sentiment of the tweet is positive.  =0: the sentiment of the tweet is neutral.  <0: the sentiment of the tweet is negative.
    `text` text, -- the text of the tweet.
    `LocationID` integer, -- location id. the location id of the poster.
    `UserID` integer, -- user id. the user id of the poster.
    primary key (TweetID),
    foreign key (LocationID) references location(LocationID),
    foreign key (UserID) references user(UserID)
);

CREATE TABLE user (
    `UserID` text, -- user id. the unique id of the user.
    `Gender` text, -- user's gender. Male / Female / Unknown.
    primary key (UserID)
);

CREATE TABLE location (
    `LocationID` integer, -- location id. unique id of the location.
    `Country` text, -- the country.
    `State` text, -- the state.
    `StateCode` text, -- state code.
    `City` text, -- the city.
    primary key (LocationID)
);

