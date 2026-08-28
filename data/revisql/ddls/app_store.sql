CREATE TABLE playstore (
    `App` text, -- Application name.
    `Category` text, -- Category the app belongs to. FAMILY 18%. GAME 11%. Other (7725) 71%.
    `Rating` real, -- Overall user rating of the app (as when scraped).
    `Reviews` integer, -- Number of user reviews for the app (as when scraped).
    `Size` text, -- Size of the app (as when scraped). Varies with device 16%. 11M 2%. Other (8948) 83%.
    `Installs` text, -- Number of user downloads/installs for the app (as when scraped). 1,000,000+ 15%. 10,000,000+ 12%. Other (8010) 74%.
    `Type` text, -- Paid or Free. Only has 'Paid' and 'Free'. Free 93%. Paid 7%.
    `Price` text, -- Price of the app (as when scraped). 0 93%. $0.99 1%. Other (653) 6%. Free means the price is 0.
    `Content Rating` text, -- Age group the app is targeted at - Children / Mature 21+ / Adult. Everyone 80%. Teen 11%. Other (919) 8%. According to Wikipedia, the different age groups are defined as:. Children: age<12~13. Teen: 13~19. Adult/Mature: 21+.
    `Genres` text, -- An app can belong to multiple genres (apart from its main category). Tools 8%. Entertainment 6%. Other (9376) 86%.

);

CREATE TABLE user_reviews (
    `App` text, -- Name of app.
    `Translated_Review` text, -- User review (Preprocessed and translated to English). nan 42%. Good 0%. Other (37185) 58%.
    `Sentiment` text, -- Overall user rating of the app (as when scraped). Positive: user holds positive attitude towards this app. Negative: user holds positive attitude / critizes on this app. Neural: user holds neural attitude. nan: user doesn't comment on this app.
    `Sentiment_Polarity` text, -- Sentiment Polarity. Sentiment polarity score. • score >= 0.5 it means pretty positive or pretty like this app. • 0 <= score < 0.5: it means user mildly likes this app. • -0.5 <= score < 0: it means this user mildly dislikes this app or slightly negative attitude. • score <-0.5: it means this user dislikes this app pretty much.
    `Sentiment_Subjectivity` text, -- Sentiment Subjectivity. Sentiment subjectivity score. more subjectivity refers to less objectivity, vice versa.

);

