CREATE TABLE countries (
   country_id             INTEGER         NOT NULL, -- example: [52769, 52770, 52771]
   country_iso_code       CHAR(2)         NOT NULL, -- example: ['SG', 'IT', 'CN']
   country_name           TEXT            NOT NULL, -- example: ['Singapore', 'Italy', 'China']
   country_subregion      TEXT            NOT NULL, -- example: ['Asia', 'Western Europe', 'Northern America']
   country_subregion_id   INTEGER         NOT NULL, -- example: [52793, 52799, 52797]
   country_region         TEXT            NOT NULL, -- example: ['Asia', 'Europe', 'Americas']
   country_region_id      INTEGER         NOT NULL, -- example: [52802, 52803, 52801]
   country_total          TEXT            NOT NULL, -- example: ['World total']
   country_total_id       INTEGER         NOT NULL, -- example: [52806]
   PRIMARY KEY (country_id)
)

CREATE TABLE customers (
   cust_id                  INTEGER         NOT NULL, -- example: [1, 2, 3]
   cust_first_name          TEXT            NOT NULL, -- example: ['Abigail', 'Anne', 'Buick']
   cust_last_name           TEXT            NOT NULL, -- example: ['Kessel', 'Koch', 'Emmerson']
   cust_gender              CHAR(1)         NOT NULL, -- example: ['M', 'F']
   cust_year_of_birth       INTEGER         NOT NULL, -- example: [1957, 1968, 1950]
   cust_marital_status      TEXT, -- example: ['single', 'married', 'divorced']
   cust_street_address      TEXT            NOT NULL, -- example: ['7 South 3rd Circle', '7 South Airway Circle', '7 South Boyd Circle']
   cust_postal_code         TEXT            NOT NULL, -- example: ['30828', '86319', '88666']
   cust_city                TEXT            NOT NULL, -- example: ['Downham Market', 'Salamanca', 'Middelburg']
   cust_city_id             INTEGER         NOT NULL, -- example: [51396, 52286, 51912]
   cust_state_province      TEXT            NOT NULL, -- example: ['England - Norfolk', 'Salamanca', 'Zeeland']
   cust_state_province_id   INTEGER         NOT NULL, -- example: [52591, 52733, 52770]
   country_id               INTEGER         NOT NULL, -- example: [52789, 52778, 52770]
   cust_main_phone_number   TEXT            NOT NULL, -- example: ['127-379-8954', '680-327-1419', '115-509-3391']
   cust_income_level        TEXT, -- example: ['G: 130,000 - 149,999', 'I: 170,000 - 189,999', 'B: 30,000 - 49,999']
   cust_credit_limit        REAL, -- example: [9000.0, 10000.0, 1500.0]
   cust_email               TEXT, -- example: ['Kessel@company.example.com', 'Koch@company.example.com', 'Emmerson@company.example.com']
   cust_total               TEXT            NOT NULL, -- example: ['Customer total']
   cust_total_id            INTEGER         NOT NULL, -- example: [52772]
   cust_src_id              INTEGER, -- example: NULL
   cust_eff_from            DATE, -- example: ['2019-01-01']
   cust_eff_to              DATE, -- example: NULL
   cust_valid               CHAR(1), -- example: ['I', 'A']
   PRIMARY KEY (cust_id),
   FOREIGN KEY (country_id) REFERENCES countries (country_id)
)

CREATE TABLE promotions (
   promo_id               INTEGER         NOT NULL, -- example: [33, 34, 35]
   promo_name             TEXT            NOT NULL, -- example: ['post promotion #20-33', 'newspaper promotion #19-34', 'TV promotion #12-35']
   promo_subcategory      TEXT            NOT NULL, -- example: ['downtown billboard', 'coupon news', 'TV commercial']
   promo_subcategory_id   INTEGER         NOT NULL, -- example: [20, 19, 12]
   promo_category         TEXT            NOT NULL, -- example: ['post', 'newspaper', 'TV']
   promo_category_id      INTEGER         NOT NULL, -- example: [9, 8, 3]
   promo_cost             REAL            NOT NULL, -- example: [77200.0, 22400.0, 61600.0]
   promo_begin_date       DATE            NOT NULL, -- example: ['2019-09-15', '2019-07-16', '2019-11-30']
   promo_end_date         DATE            NOT NULL, -- example: ['2019-11-15', '2019-09-16', '2020-01-30']
   promo_total            TEXT            NOT NULL, -- example: ['Promotion total']
   promo_total_id         INTEGER         NOT NULL, -- example: [1]
   PRIMARY KEY (promo_id)
)

CREATE TABLE products (
   prod_id                 INTEGER         NOT NULL, -- example: [14, 19, 21]
   prod_name               TEXT            NOT NULL, -- example: ['Pitching Machine and Batting Cage Combo', 'Cricket Bat Bag', 'Speed Trainer Bats and Training Program']
   prod_desc               TEXT            NOT NULL, -- example: ['Pitching Machine and Batting Cage Combo', 'Cricket bat bag', 'Speed Trainer Bats and Training Program']
   prod_subcategory        TEXT            NOT NULL, -- example: ['Training Aids and Equipment', 'Cricket Bat', 'Baseballs']
   prod_subcategory_id     INTEGER         NOT NULL, -- example: [2035, 2051, 2031]
   prod_subcategory_desc   TEXT            NOT NULL, -- example: ['Training Aids and Equipment', 'Cricket Bat', 'Baseballs']
   prod_category           TEXT            NOT NULL, -- example: ['Baseball', 'Cricket']
   prod_category_id        INTEGER         NOT NULL, -- example: [203, 205]
   prod_category_desc      TEXT            NOT NULL, -- example: ['Baseball', 'Cricket']
   prod_weight_class       INTEGER         NOT NULL, -- example: [1]
   prod_unit_of_measure    TEXT, -- example: ['U']
   prod_pack_size          TEXT            NOT NULL, -- example: ['P']
   supplier_id             INTEGER         NOT NULL, -- example: [1]
   prod_status             TEXT            NOT NULL, -- example: ['STATUS']
   prod_list_price         REAL            NOT NULL, -- example: [999.99, 55.99, 899.99]
   prod_min_price          REAL            NOT NULL, -- example: [999.99, 55.99, 899.99]
   prod_total              TEXT            NOT NULL, -- example: ['TOTAL']
   prod_total_id           INTEGER         NOT NULL, -- example: [1]
   prod_src_id             INTEGER, -- example: NULL
   prod_eff_from           DATE, -- example: ['2019-01-01 00:00:00']
   prod_eff_to             DATE, -- example: NULL
   prod_valid              CHAR(1), -- example: ['A']
   PRIMARY KEY (prod_id)
)

CREATE TABLE times (
   time_id                   DATE          NOT NULL, -- example: ['2019-01-01', '2019-01-02', '2019-01-03']
   day_name                  TEXT          NOT NULL, -- example: ['Friday', 'Saturday', 'Sunday']
   day_number_in_week        INTEGER       NOT NULL, -- example: [5, 6, 7]
   day_number_in_month       INTEGER       NOT NULL, -- example: [31, 1, 2]
   calendar_week_number      INTEGER       NOT NULL, -- example: [22, 23, 24]
   fiscal_week_number        INTEGER       NOT NULL, -- example: [22, 23, 24]
   week_ending_day           DATE          NOT NULL, -- example: ['2019-06-02', '2019-06-09', '2019-06-16']
   week_ending_day_id        INTEGER       NOT NULL, -- example: [1670, 1506, 1554]
   calendar_month_number     INTEGER       NOT NULL, -- example: [5, 6, 7]
   fiscal_month_number       INTEGER       NOT NULL, -- example: [5, 6, 7]
   calendar_month_desc       TEXT          NOT NULL, -- example: ['2019-05', '2019-06', '2019-07']
   calendar_month_id         INTEGER       NOT NULL, -- example: [1676, 1677, 1678]
   fiscal_month_desc         TEXT          NOT NULL, -- example: ['2019-05', '2019-06', '2019-07']
   fiscal_month_id           INTEGER       NOT NULL, -- example: [1724, 1725, 1726]
   days_in_cal_month         INTEGER       NOT NULL, -- example: [31, 30, 28]
   days_in_fis_month         INTEGER       NOT NULL, -- example: [35, 28, 25]
   end_of_cal_month          DATE          NOT NULL, -- example: ['2019-05-31', '2019-06-30', '2019-07-31']
   end_of_fis_month          DATE          NOT NULL, -- example: ['2019-05-31', '2019-06-28', '2019-07-26']
   calendar_month_name       TEXT          NOT NULL, -- example: ['May', 'June', 'July']
   fiscal_month_name         TEXT          NOT NULL, -- example: ['May', 'June', 'July']
   calendar_quarter_desc     CHAR(7)       NOT NULL, -- example: ['2019-02', '2019-03', '2019-01']
   calendar_quarter_id       INTEGER       NOT NULL, -- example: [1770, 1771, 1769]
   fiscal_quarter_desc       CHAR(7)       NOT NULL, -- example: ['2019-02', '2019-03', '2019-01']
   fiscal_quarter_id         INTEGER       NOT NULL, -- example: [1786, 1787, 1785]
   days_in_cal_quarter       INTEGER       NOT NULL, -- example: [91, 92, 90]
   days_in_fis_quarter       INTEGER       NOT NULL, -- example: [91, 88, 98]
   end_of_cal_quarter        DATE          NOT NULL, -- example: ['2019-06-30', '2019-09-30', '2019-03-31']
   end_of_fis_quarter        DATE          NOT NULL, -- example: ['2019-06-28', '2019-09-27', '2019-03-29']
   calendar_quarter_number   INTEGER       NOT NULL, -- example: [2, 3, 1]
   fiscal_quarter_number     INTEGER       NOT NULL, -- example: [2, 3, 1]
   calendar_year             INTEGER       NOT NULL, -- example: [2019, 2020, 2021]
   calendar_year_id          INTEGER       NOT NULL, -- example: [1802, 1803, 1804]
   fiscal_year               INTEGER       NOT NULL, -- example: [2019, 2020, 2021]
   fiscal_year_id            INTEGER       NOT NULL, -- example: [1806, 1807, 1808]
   days_in_cal_year          INTEGER       NOT NULL, -- example: [365, 366]
   days_in_fis_year          INTEGER       NOT NULL, -- example: [361, 364, 371]
   end_of_cal_year           DATE          NOT NULL, -- example: ['2019-12-31', '2020-12-31', '2021-12-31']
   end_of_fis_year           DATE          NOT NULL, -- example: ['2019-12-27', '2020-12-26', '2021-12-31']
   PRIMARY KEY (time_id)
)

CREATE TABLE channels (
   channel_id         INTEGER         NOT NULL, -- example: [2, 3, 4]
   channel_desc       TEXT            NOT NULL, -- example: ['Partners', 'Direct Sales', 'Internet']
   channel_class      TEXT            NOT NULL, -- example: ['Others', 'Direct', 'Indirect']
   channel_class_id   INTEGER         NOT NULL, -- example: [14, 12, 13]
   channel_total      TEXT            NOT NULL, -- example: ['Channel total']
   channel_total_id   INTEGER         NOT NULL, -- example: [1]
   PRIMARY KEY (channel_id)
)

CREATE TABLE sales (
   prod_id         INTEGER         NOT NULL, -- example: [13, 14, 15]
   cust_id         INTEGER         NOT NULL, -- example: [987, 1660, 1762]
   time_id         DATE            NOT NULL, -- example: ['2019-01-10', '2019-01-20', '2019-01-30']
   channel_id      INTEGER         NOT NULL, -- example: [3, 2, 4]
   promo_id        INTEGER         NOT NULL, -- example: [999, 33, 350]
   quantity_sold   INTEGER         NOT NULL, -- example: [1]
   amount_sold     REAL            NOT NULL, -- example: [1232.16, 1205.99, 1237.31]
   FOREIGN KEY (promo_id)   REFERENCES promotions (promo_id),
   FOREIGN KEY (cust_id)    REFERENCES customers (cust_id),
   FOREIGN KEY (prod_id)    REFERENCES products (prod_id),
   FOREIGN KEY (channel_id) REFERENCES channels (channel_id),
   FOREIGN KEY (time_id) REFERENCES times (time_id)
)

CREATE TABLE costs (
   prod_id      INTEGER         NOT NULL, -- example: [13, 14, 15]
   time_id      DATE            NOT NULL, -- example: ['2019-02-10', '2019-01-19', '2019-02-02']
   promo_id     INTEGER         NOT NULL, -- example: [350, 351, 999]
   channel_id   INTEGER         NOT NULL, -- example: [2, 3, 4]
   unit_cost    REAL            NOT NULL, -- example: [813.07, 886.45, 863.64]
   unit_price   REAL            NOT NULL, -- example: [1237.31, 1108.99, 1259.99]
   FOREIGN KEY (promo_id)   REFERENCES promotions (promo_id),
   FOREIGN KEY (prod_id)    REFERENCES products (prod_id),
   FOREIGN KEY (time_id)    REFERENCES times (time_id),
   FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
)

CREATE TABLE supplementary_demographics (
   cust_id                   INTEGER           NOT NULL, -- example: [100001, 100002, 100003]
   education                 TEXT, -- example: ['< Bach.', 'Bach.', 'Assoc-A']
   occupation                TEXT, -- example: ['Exec.', 'Prof.', 'Sales']
   household_size            TEXT, -- example: ['2', '3', '9+']
   yrs_residence             INTEGER, -- example: [3, 4, 6]
   affinity_card             INTEGER, -- example: [0, 1]
   cricket                   INTEGER, -- example: [0, 1]
   baseball                  INTEGER, -- example: [0, 1]
   tennis                    INTEGER, -- example: [1, 0]
   soccer                    INTEGER, -- example: [1, 0]
   golf                      INTEGER, -- example: [1]
   unknown                   INTEGER, -- example: [0, 1]
   misc                      INTEGER, -- example: [0, 1]
   comments                  TEXT, -- example (truncated): ['Thanks a lot for my new affinity card. I love the discounts and have since started shopping at your ...', 'The more times that I shop at your store, the more times I am impressed.  Don''t change anything', 'It is a good way to attract new shoppers. After shopping at your store for more than a month, I am r...']
   PRIMARY KEY (cust_id)
)

CREATE TABLE currency (
   country TEXT, -- example: ['Singapore', 'Italy', 'China']
   year INTEGER, -- example: [2019, 2020, 2021]
   month INTEGER, -- example: [5, 6, 7]
   to_us REAL -- example: [1.0, 0.74]
)

