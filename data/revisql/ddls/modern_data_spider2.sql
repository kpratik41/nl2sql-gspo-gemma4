CREATE TABLE "pizza_names" (
"pizza_id" INTEGER, -- example: [1, 2]
  "pizza_name" TEXT -- example: ['Meatlovers', 'Vegetarian']
)

CREATE TABLE "companies_funding" (
"company_id" INTEGER, -- example: [548, 645, 615]
  "valuation" INTEGER, -- example: [2000000000, 1000000000, 10000000000]
  "funding" INTEGER, -- example: [449000000, 188000000, 2000000000]
  "select_investors" TEXT -- example: ['"Accel Partners, Index Ventures, Insight Venture Partners"', '"Sequoia Capital China, China Life Investment Holding Company, Qiming Venture Partners"', '"CreditEase Fintech Investment Fund, BMW i Ventures, SoftBank Group"']
)

CREATE TABLE "pizza_customer_orders" (
"order_id" INTEGER, -- example: [1, 2, 3]
  "customer_id" INTEGER, -- example: [101, 102, 103]
  "pizza_id" INTEGER, -- example: [1, 2]
  "exclusions" TEXT, -- example: ['4', '2,6']
  "extras" TEXT, -- example: ['1', '1,5', '1,4']
  "order_time" TEXT -- example: ['2021-01-01 18:05:02', '2021-01-01 19:00:52', '2021-01-02 23:51:23']
)

CREATE TABLE "pizza_toppings" (
"topping_id" INTEGER, -- example: [1, 2, 3]
  "topping_name" TEXT -- example: ['Bacon', 'BBQ Sauce', 'Beef']
)

CREATE TABLE "trees" (
"idx" INTEGER, -- example: [199121, 32277, 164564]
  "tree_id" INTEGER, -- example: [414328, 155915, 362104]
  "tree_dbh" INTEGER, -- example: [4, 2, 23]
  "stump_diam" INTEGER, -- example: [0, 4, 12]
  "status" TEXT, -- example: ['Alive', 'Stump', 'Dead']
  "health" TEXT, -- example: ['Good', 'Fair', 'Poor']
  "spc_latin" TEXT, -- example: ['Ulmus americana', 'Eucommia ulmoides', 'Tilia cordata']
  "spc_common" TEXT, -- example: ['American elm', 'hardy rubber tree', 'littleleaf linden']
  "address" TEXT, -- example: ['1301 RYAWA AVENUE', '506 BEACH 69 STREET', '2312 WESTERVELT AVENUE']
  "zipcode" INTEGER, -- example: [10474, 11692, 10469]
  "borocode" INTEGER, -- example: [2, 4, 3]
  "boroname" TEXT, -- example: ['Bronx', 'Queens', 'Brooklyn']
  "nta_name" TEXT, -- example: ['Hunts Point', 'Hammels-Arverne-Edgemere', 'Allerton-Pelham Gardens']
  "state" TEXT, -- example: ['New York']
  "latitude" REAL, -- example: [40.80504923, 40.5949501, 40.85977962]
  "longitude" REAL -- example: [-73.88385512, -73.79834048, -73.8399055]
)

CREATE TABLE "pizza_recipes" (
"pizza_id" INTEGER, -- example: [1, 2]
  "toppings" TEXT -- example: ['1, 2, 3, 4, 5, 6, 8, 10', '4, 6, 7, 9, 11, 12']
)

CREATE TABLE "statistics" (
"date" TEXT, -- example: ['2020-05-27 00:00:00', '2020-06-26 00:00:00', '2020-09-05 00:00:00']
  "state" TEXT, -- example: ['NC', 'CO', 'MA']
  "total_cases" INTEGER, -- example: [24628, 58818, 58989]
  "total_deaths" INTEGER -- example: [794, 1303, 1971]
)

CREATE TABLE "income_trees" (
"zipcode" INTEGER, -- example: [11205, 11218, 10451]
  "Estimate_Total" INTEGER, -- example: [15198, 24909, 18140]
  "Margin_of_Error_Total" INTEGER, -- example: [353, 371, 405]
  "Estimate_Median_income" INTEGER, -- example: [47575, 56120, 26048]
  "Margin_of_Error_Median_income" INTEGER, -- example: [3834, 3925, 2140]
  "Estimate_Mean_income" INTEGER, -- example: [73353, 78208, 40836]
  "Margin_of_Error_Mean_income" INTEGER -- example: [3929, 3788, 3424]
)

CREATE TABLE "pizza_clean_runner_orders" (
"order_id" INTEGER, -- example: [1, 2, 3]
  "runner_id" INTEGER, -- example: [1, 2, 3]
  "pickup_time" TEXT, -- example: ['2021-01-01 18:15:34', '2021-01-01 19:10:54', '2021-01-03 00:12:37']
  "distance" REAL, -- example: [20.0, 13.4, 23.4]
  "duration" REAL, -- example: [32.0, 27.0, 20.0]
  "cancellation" TEXT -- example: ['Restaurant Cancellation', 'Customer Cancellation']
)

CREATE TABLE "pizza_runner_orders" (
"order_id" INTEGER, -- example: [1, 2, 3]
  "runner_id" INTEGER, -- example: [1, 2, 3]
  "pickup_time" TEXT, -- example: ['2021-01-01 18:15:34', '2021-01-01 19:10:54', '2021-01-03 00:12:37']
  "distance" TEXT, -- example: ['20km', '13.4km', '23.4']
  "duration" TEXT, -- example: ['32 minutes', '27 minutes', '20 mins']
  "cancellation" TEXT -- example: ['Restaurant Cancellation', 'Customer Cancellation']
)

CREATE TABLE "word_list" (
"words" TEXT -- example: ['cannach', 'ouistitis', 'revacate']
)

CREATE TABLE "companies_dates" (
"company_id" INTEGER, -- example: [109, 821, 153]
  "date_joined" TEXT, -- example: ['2020-09-08T00:00:00.000', '2019-05-16T00:00:00.000', '2019-07-11T00:00:00.000']
  "year_founded" INTEGER -- example: [2004, 2009, 2016]
)

CREATE TABLE "pizza_get_extras" (
"row_id" INTEGER, -- example: [1, 2]
  "order_id" INTEGER, -- example: [5, 7, 9]
  "extras" INTEGER, -- example: [1, 5, 4]
  "extras_count" INTEGER -- example: [1, 2]
)

CREATE TABLE "pizza_get_exclusions" (
"row_id" INTEGER, -- example: [1, 2]
  "order_id" INTEGER, -- example: [4, 9, 10]
  "exclusions" INTEGER, -- example: [4, 2, 6]
  "total_exclusions" INTEGER -- example: [3, 1, 2]
)

CREATE TABLE "pizza_clean_customer_orders" (
"order_id" INTEGER, -- example: [1, 2, 3]
  "customer_id" INTEGER, -- example: [101, 102, 103]
  "pizza_id" INTEGER, -- example: [1, 2]
  "exclusions" TEXT, -- example: ['4', '2,6']
  "extras" TEXT, -- example: ['1', '1,5', '1,4']
  "order_time" TEXT -- example: ['2021-01-01 18:05:02', '2021-01-01 19:00:52', '2021-01-02 23:51:23']
)

CREATE TABLE "companies_industries" (
"company_id" INTEGER, -- example: [316, 162, 803]
  "industry" TEXT -- example: ['Fintech', 'Internet software & services', 'E-commerce & direct-to-consumer']
)

CREATE TABLE "pizza_runners" (
"runner_id" INTEGER, -- example: [1, 2, 3]
  "registration_date" TEXT -- example: ['2021-01-01', '2021-01-03', '2021-01-08']
)

