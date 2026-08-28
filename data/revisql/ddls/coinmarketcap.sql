CREATE TABLE coins (
    `id` integer, -- unique id number identifying coins.
    `name` text, -- name of coins.
    `slug` text, -- slug names of coins.
    `symbol` text, -- symbol of names of coins.
    `status` text, -- status of coins. commonsense reasoning:. • active: this coin is active and available for transactions. • untracked: this coin cannot be tracked. • inactive: this coin is not available for transactions. • extinct: this coin has been disappeared.
    `category` text, -- category of coins. commonsense reasoning:. • token:. • coin:.
    `description` text, -- description of coins.
    `subreddit` text, -- name of coins on reddit.
    `notice` text, -- notice if any.
    `tags` text, -- tags of coins.
    `tag_names` text, -- tag names of coins.
    `website` text, -- website introduction of coins.
    `platform_id` integer, -- platform id. id number of the platform.
    `date_added` text, -- date added. the added date.
    `date_launched` text, -- date lanched. lanched date.
    primary key (id)
);

CREATE TABLE historical (
    `date` date, -- transaction date.
    `coin_id` integer, -- id number referring to coins.
    `cmc_rank` integer, -- coinmarketcap rank. CMC rank is a metric used by CoinMarketCap to rank the relative popularity of different cryptocurrencies.
    `market_cap` real, -- Market capitalization. Market capitalization refers to how much a coin is worth as determined by the coin market. dollars. market cap (latest trade price x circulating supply).
    `price` real, -- how much is a coin. dollars.
    `open` real, -- the price at which a cryptocurrency (coin) opens at a time period. dollars. commonsense reasoning:. if null or empty, it means this coins has not opened yet today.
    `high` real, -- highest price. dollars.
    `low` real, -- lowest price. commonsense reasoning:. 1. It's the best time to purchase this coin. 2. users can make the max profit today by computation: high - low.
    `close` real, -- the price at which a cryptocurrency (coin) closes at a time period, for example at the end of the day.
    `time_high` text, -- the time to achieve highest price.
    `time_low` text, -- the time to achieve lowest price.
    `volume_24h` real, -- the total value of cryptocurrency coined traded in the past 24 hours.
    `percent_change_1h` real, -- percent change 1h. price percentage change in the first hour. The change is the difference (in percent) between the price now compared to the price around this time 1 hours ago.
    `percent_change_24h` real, -- percent change 24h. price percentage change in the first 24 hours. The change is the difference (in percent) between the price now compared to the price around this time 24 hours ago.
    `percent_change_7d` real, -- percent change 7d. price percentage change in the first 7 days. The change is the difference (in percent) between the price now compared to the price around this time 7 days ago.
    `circulating_supply` real, -- circulating supply. the best approximation of the number of coins that are circulating in the market and in the general public's hands.
    `total_supply` real, -- total supply. the total amount of coins in existence right now.
    `max_supply` real, -- max supply. Max Supply is the best approximation of the maximum amount of coins that will ever exist in the lifetime of the cryptocurrency. commonsense reasoning:. the number of coins verifiably burned so far = max_supply - total_supply.
    `num_market_pairs` integer, -- number market pairs. number of market pairs across all exchanges trading each currency. commonsense reasoning:. active coins that have gone inactive can easily be identified as having a num_market_pairs of 0.

);

