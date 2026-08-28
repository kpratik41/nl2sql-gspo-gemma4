CREATE TABLE "weekly_sales" (
"week_date" TEXT, -- example: ['31/8/20', '24/8/20', '17/8/20']
  "region" TEXT, -- example: ['ASIA', 'USA', 'EUROPE']
  "platform" TEXT, -- example: ['Retail', 'Shopify']
  "segment" TEXT, -- example: ['C3', 'F1', 'C1']
  "customer_type" TEXT, -- example: ['New', 'Guest', 'Existing']
  "transactions" INTEGER, -- example: [120631, 31574, 529151]
  "sales" INTEGER -- example: [3656163, 996575, 16509610]
)

CREATE TABLE "shopping_cart_users" (
"user_id" INTEGER, -- example: [1, 2, 3]
  "cookie_id" TEXT, -- example: ['c4ca42', 'c81e72', 'eccbc8']
  "start_date" TEXT -- example: ['2020-02-04', '2020-01-18', '2020-02-21']
)

CREATE TABLE "bitcoin_members" (
"member_id" TEXT, -- example: ['c4ca42', 'c81e72', 'eccbc8']
  "first_name" TEXT, -- example: ['Danny', 'Vipul', 'Charlie']
  "region" TEXT -- example: ['Australia', 'United States', 'Africa']
)

CREATE TABLE "interest_metrics" (
"_month" REAL, -- example: [7.0, 8.0, 9.0]
  "_year" REAL, -- example: [2018.0, 2019.0]
  "month_year" TEXT, -- example: ['07-2018', '08-2018', '09-2018']
  "interest_id" REAL, -- example: [32486.0, 6106.0, 18923.0]
  "composition" REAL, -- example: [11.89, 9.93, 10.85]
  "index_value" REAL, -- example: [6.19, 5.31, 5.29]
  "ranking" INTEGER, -- example: [1, 2, 3]
  "percentile_ranking" REAL -- example: [99.86, 99.73, 99.59]
)

CREATE TABLE "customer_regions" (
"region_id" INTEGER, -- example: [1, 2, 3]
  "region_name" TEXT -- example: ['Australia', 'America', 'Africa']
)

CREATE TABLE "customer_transactions" (
"customer_id" INTEGER, -- example: [429, 155, 398]
  "txn_date" TEXT, -- example: ['2020-01-21', '2020-01-10', '2020-01-01']
  "txn_type" TEXT, -- example: ['deposit', 'withdrawal', 'purchase']
  "txn_amount" INTEGER -- example: [82, 712, 196]
)

CREATE TABLE "bitcoin_transactions" (
"txn_id" INTEGER, -- example: [1, 2, 3]
  "member_id" TEXT, -- example: ['c81e72', 'eccbc8', 'a87ff6']
  "ticker" TEXT, -- example: ['BTC', 'ETH']
  "txn_date" TEXT, -- example: ['01-01-2017', '02-01-2017', '03-01-2017']
  "txn_type" TEXT, -- example: ['BUY', 'SELL']
  "quantity" REAL, -- example: [50.0, 9.562185136, 8.842987018]
  "percentage_fee" REAL, -- example: [0.3, 0.0, 0.02]
  "txn_time" TEXT -- example: ['2017-01-01T00:00:00.000Z', '2017-01-01T01:22:32.097Z', '2017-01-01T06:22:20.203Z']
)

CREATE TABLE "customer_nodes" (
"customer_id" INTEGER, -- example: [1, 2, 3]
  "region_id" INTEGER, -- example: [3, 5, 1]
  "node_id" INTEGER, -- example: [4, 5, 3]
  "start_date" TEXT, -- example: ['2020-01-02', '2020-01-03', '2020-01-27']
  "end_date" TEXT -- example: ['2020-01-03', '2020-01-17', '2020-02-18']
)

CREATE TABLE "cleaned_weekly_sales" (
"week_date_formatted" TEXT, -- example: ['2020-8-31', '2020-8-24', '2020-8-17']
  "week_date" TEXT, -- example: ['2020-08-31', '2020-08-24', '2020-08-17']
  "region" TEXT, -- example: ['ASIA', 'USA', 'EUROPE']
  "platform" TEXT, -- example: ['Retail', 'Shopify']
  "segment" TEXT, -- example: ['C3', 'F1', 'unknown']
  "customer_type" TEXT, -- example: ['New', 'Guest', 'Existing']
  "transactions" INTEGER, -- example: [120631, 31574, 529151]
  "sales" INTEGER, -- example: [3656163, 996575, 16509610]
  "week_number" INTEGER, -- example: [36, 35, 34]
  "month_number" INTEGER, -- example: [8, 7, 6]
  "calendar_year" INTEGER, -- example: [2020, 2019, 2018]
  "age_band" TEXT, -- example: ['Retirees', 'Young Adults', 'unknown']
  "demographic" TEXT, -- example: ['Couples', 'Families', 'unknown']
  "avg_transaction" REAL -- example: [30.31, 31.56, 31.2]
)

CREATE TABLE "veg_txn_df" (
"index" INTEGER, -- example: [0, 1, 2]
  "txn_date" TEXT, -- example: ['2020-07-01 00:00:00', '2020-07-02 00:00:00', '2020-07-03 00:00:00']
  "txn_time" TEXT, -- example: ['09:15:07', '09:17:27', '09:17:33']
  "item_code" INTEGER, -- example: [102900005117056, 102900005115960, 102900005115823]
  "qty_sold(kg)" REAL, -- example: [0.396, 0.849, 0.409]
  "unit_selling_px_rmb/kg" REAL, -- example: [7.6, 3.2, 10.0]
  "sale/return" TEXT, -- example: ['sale', 'return']
  "discount(%)" INTEGER, -- example: [1]
  "day_of_week" TEXT -- example: ['Wednesday', 'Thursday', 'Friday']
)

CREATE TABLE "shopping_cart_events" (
"visit_id" TEXT, -- example: ['ccf365', 'd58cbd', '9a2f24']
  "cookie_id" TEXT, -- example: ['c4ca42', 'c81e72', 'eccbc8']
  "page_id" INTEGER, -- example: [1, 2, 6]
  "event_type" INTEGER, -- example: [1, 2, 3]
  "sequence_number" INTEGER, -- example: [1, 2, 3]
  "event_time" TEXT -- example: ['2020-02-04 19:16:09.182546', '2020-02-04 19:16:17.358191', '2020-02-04 19:16:58.454669']
)

CREATE TABLE "shopping_cart_page_hierarchy" (
"page_id" INTEGER, -- example: [1, 2, 3]
  "page_name" TEXT, -- example: ['Home Page', 'All Products', 'Salmon']
  "product_category" TEXT, -- example: ['Fish', 'Luxury', 'Shellfish']
  "product_id" REAL -- example: [1.0, 2.0, 3.0]
)

CREATE TABLE "bitcoin_prices" (
"ticker" TEXT, -- example: ['ETH', 'BTC']
  "market_date" TEXT, -- example: ['29-08-2021', '28-08-2021', '27-08-2021']
  "price" REAL, -- example: [3177.84, 3243.9, 3273.58]
  "open" REAL, -- example: [3243.96, 3273.78, 3093.78]
  "high" REAL, -- example: [3282.21, 3284.58, 3279.93]
  "low" REAL, -- example: [3162.79, 3212.24, 3063.37]
  "volume" TEXT, -- example: ['582.04K', '466.21K', '839.54K']
  "change" TEXT -- example: ['-2.04%', '-0.91%', '5.82%']
)

CREATE TABLE "interest_map" (
"id" INTEGER, -- example: [1, 2, 3]
  "interest_name" TEXT, -- example: ['Fitness Enthusiasts', 'Gamers', 'Car Enthusiasts']
  "interest_summary" TEXT, -- example: ['Consumers using fitness tracking apps and websites.', 'Consumers researching game reviews and cheat codes.', 'Readers of automotive news and car reviews.']
  "created_at" TEXT, -- example: ['2016-05-26 14:57:59', '2016-06-09 16:28:11', '2016-08-11 12:08:56']
  "last_modified" TEXT -- example: ['2018-05-23 11:30:12', '2018-05-23 11:30:13', '2018-03-16 13:14:00']
)

CREATE TABLE "veg_loss_rate_df" (
"index" INTEGER, -- example: [0, 1, 2]
  "item_code" INTEGER, -- example: [102900005115168, 102900005115199, 102900005115250]
  "item_name" TEXT, -- example: ['Niushou Shengcai', 'Sichuan Red Cedar', 'Xixia Black Mushroom (1)']
  "loss_rate_%" REAL -- example: [4.39, 10.46, 10.8]
)

CREATE TABLE "shopping_cart_campaign_identifier" (
"campaign_id" INTEGER, -- example: [1, 2, 3]
  "products" TEXT, -- example: ['1-3', '4-5', '6-8']
  "campaign_name" TEXT, -- example: ['BOGOF - Fishing For Compliments', '25% Off - Living The Lux Life', 'Half Off - Treat Your Shellf(ish)']
  "start_date" TEXT, -- example: ['2020-01-01', '2020-01-15', '2020-02-01']
  "end_date" TEXT -- example: ['2020-01-14', '2020-01-28', '2020-03-31']
)

CREATE TABLE "veg_cat" (
"index" INTEGER, -- example: [0, 1, 2]
  "item_code" INTEGER, -- example: [102900005115168, 102900005115199, 102900005115625]
  "item_name" TEXT, -- example: ['Niushou Shengcai', 'Sichuan Red Cedar', 'Local Xiaomao Cabbage']
  "category_code" INTEGER, -- example: [1011010101, 1011010201, 1011010402]
  "category_name" TEXT -- example: ['Flower/Leaf Vegetables', 'Cabbage', 'Aquatic Tuberous Vegetables']
)

CREATE TABLE "veg_whsle_df" (
"index" INTEGER, -- example: [0, 1, 2]
  "whsle_date" TEXT, -- example: ['2020-07-01 00:00:00', '2020-07-02 00:00:00', '2020-07-03 00:00:00']
  "item_code" INTEGER, -- example: [102900005115762, 102900005115779, 102900005115786]
  "whsle_px_rmb-kg" REAL -- example: [3.88, 6.72, 3.19]
)

CREATE TABLE "shopping_cart_event_identifier" (
"event_type" INTEGER, -- example: [1, 2, 3]
  "event_name" TEXT -- example: ['Page View', 'Add to Cart', 'Purchase']
)

