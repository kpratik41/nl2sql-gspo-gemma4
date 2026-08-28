CREATE TABLE mst_users(
    user_id         varchar(255) -- example: ['U001', 'U002', 'U003']
  , sex             varchar(255) -- example: ['M', 'F']
  , birth_date      varchar(255) -- example: ['1977-06-17', '1953-06-12', '1965-01-06']
  , register_date   varchar(255) -- example: ['2016-10-01', '2016-10-05', '2016-10-10']
  , register_device varchar(255) -- example: ['pc', 'sp', 'app']
  , withdraw_date   varchar(255) -- example: ['2016-10-10']
)

CREATE TABLE action_log(
    session  varchar(255) -- example: ['989004ea', '47db0370', '87b5725f']
  , user_id  varchar(255) -- example: ['U001', 'U002']
  , action   varchar(255) -- example: ['purchase', 'view', 'favorite']
  , category varchar(255) -- example: ['drama', 'action']
  , products varchar(255) -- example: ['D001,D002', 'D001', 'D002']
  , amount   integer -- example: [2000, 1000]
  , stamp    varchar(255) -- example: ['2016-11-03 18:10:00', '2016-11-03 18:00:00', '2016-11-03 18:01:00']
)

CREATE TABLE activity_log(
    stamp        varchar(255) -- example: ['2017-01-09 12:18:43', '2017-01-09 12:19:27', '2017-01-09 12:20:03']
  , session      varchar(255) -- example: ['989004ea', '47db0370', '1cf7678e']
  , action       varchar(255) -- example: ['view']
  , option       varchar(255) -- example: ['search', 'page', 'detail']
  , path         varchar(255) -- example: ['/search_list/', '/search_input/', '/detail/']
  , search_type  varchar(255) -- example: ['Area-L-with-Job', 'Pref', 'Area-S']
)

CREATE TABLE read_log(
    stamp        varchar(255) -- example: ['2016-12-29 21:45:47', '2016-12-29 21:45:56', '2016-12-29 21:46:05']
  , session      varchar(255) -- example: ['afbd3d09', 'df6eb25d', '77d477cc']
  , action       varchar(255) -- example: ['view', 'read-20%', 'read-40%']
  , url          varchar(255) -- example: ['http://www.example.com/article?id=news341', 'http://www.example.com/article?id=news731', 'http://www.example.com/article?id=it605']
)

CREATE TABLE form_log(
    stamp    varchar(255) -- example: ['2016-12-30 00:56:08', '2016-12-30 00:57:04', '2016-12-30 00:57:56']
  , session  varchar(255) -- example: ['647219c7', '9b5f320f', '8e9afadc']
  , action   varchar(255) -- example: ['view']
  , path     varchar(255) -- example: ['/regist/input', '/cart/input', '/regist/confirm']
  , status   varchar(255) -- example: ['error']
)

CREATE TABLE form_error_log(
    stamp       varchar(255) -- example: ['2016-12-30 00:56:08', '2016-12-30 00:57:21', '2016-12-30 00:56:09']
  , session     varchar(255) -- example: ['004dc3ef', '00700be4', '01061716']
  , form        varchar(255) -- example: ['regist', 'cart']
  , field       varchar(255) -- example: ['email', 'kana', 'zip']
  , error_type  varchar(255) -- example: ['require', 'format_error', 'not_kana']
  , value       varchar(255) -- example: ['101-', 'xxx---.co.jp', 'xxx@---cojp']
)

CREATE TABLE action_log_with_ip(
    session  varchar(255) -- example: ['0CVKaz', '1QceiB', '1hI43A']
  , user_id  varchar(255) -- example: ['U001', 'U002', 'U003']
  , action   varchar(255) -- example: ['view']
  , ip       varchar(255) -- example: ['216.58.220.238', '98.139.183.24', '210.154.149.63']
  , stamp    varchar(255) -- example: ['2016-11-03 18:00:00', '2016-11-03 19:00:00', '2016-11-03 20:00:00']
)

CREATE TABLE access_log(
    session  varchar(255) -- example: ['98900e', '1cf768', '87b575']
  , user_id  varchar(255) -- example: ['U001', 'U002', '0CVKaz']
  , action   varchar(255) -- example: ['view', '1CwlSX', '3JMO2k']
  , stamp    varchar(255) -- example: ['2016-01-01 18:00:00', '2016-01-02 20:00:00', '2016-01-03 22:00:00']
)

CREATE TABLE action_log_with_noise(
    stamp       varchar(255) -- example: NULL
  , session     varchar(255) -- example: NULL
  , action      varchar(255) -- example: NULL
  , products    varchar(255) -- example: NULL
  , url         text -- example: NULL
  , ip          varchar(255) -- example: NULL
  , user_agent  text -- example: NULL
)

CREATE TABLE invalid_action_log(
    stamp     varchar(255) -- example: ['2016-11-03 18:10:00', '2016-11-03 18:00:00', '2016-11-03 18:01:00']
  , session   varchar(255) -- example: ['0CVKaz', '1QceiB']
  , user_id   varchar(255) -- example: ['U001', 'U002']
  , action    varchar(255) -- example: ['purchase', 'favorite', 'view']
  , category  varchar(255) -- example: ['drama', 'action']
  , products  varchar(255) -- example: ['D001,D002', 'D001', 'D002']
  , amount    integer -- example: [2000, 1000]
)

CREATE TABLE mst_categories(
    id     integer -- example: [1, 2, 3]
  , name   varchar(255) -- example: ['ladys_fashion', 'mens_fashion', 'book']
  , stamp  varchar(255) -- example: ['2016-01-01 10:00:00', '2016-02-01 10:00:00']
)

CREATE TABLE dup_action_log(
    stamp     varchar(255) -- example: ['2016-11-03 18:00:00', '2016-11-03 19:00:00', '2016-11-03 20:00:00']
  , session   varchar(255) -- example: ['989004ea', '47db0370', '1cf7678e']
  , user_id   varchar(255) -- example: ['U001', 'U002', 'U003']
  , action    varchar(255) -- example: ['click']
  , products  varchar(255) -- example: ['D001', 'D002', 'A001']
)

CREATE TABLE mst_products_20161201(
    product_id  varchar(255) -- example: ['A001', 'A002', 'B001']
  , name        varchar(255) -- example: ['AAA', 'AAB', 'BBB']
  , price       integer -- example: [3000, 4000, 5000]
  , updated_at  varchar(255) -- example: ['2016-11-03 18:00:00', '2016-11-03 19:00:00', '2016-11-03 20:00:00']
)

CREATE TABLE mst_products_20170101(
    product_id  varchar(255) -- example: ['A001', 'A002', 'B002']
  , name        varchar(255) -- example: ['AAA', 'AAB', 'BBD']
  , price       integer -- example: [3000, 4000, 5000]
  , updated_at  varchar(255) -- example: ['2016-11-03 18:00:00', '2016-11-03 19:00:00', '2016-11-03 21:00:00']
)

CREATE TABLE app1_mst_users (
    user_id varchar(255) -- example: ['U001', 'U002']
  , name    varchar(255) -- example: ['Sato', 'Suzuki']
  , email   varchar(255) -- example: ['sato@example.com', 'suzuki@example.com']
)

CREATE TABLE app2_mst_users (
    user_id varchar(255) -- example: ['U001', 'U002']
  , name    varchar(255) -- example: ['Ito', 'Tanaka']
  , phone   varchar(255) -- example: ['080-xxxx-xxxx', '070-xxxx-xxxx']
)

CREATE TABLE mst_users_with_card_number (
    user_id     varchar(255) -- example: ['U001', 'U002', 'U003']
  , card_number varchar(255) -- example: ['1234-xxxx-xxxx-xxxx', '5678-xxxx-xxxx-xxxx']
)

CREATE TABLE purchase_log (
    purchase_id integer -- example: [10001, 10002, 10003]
  , user_id     varchar(255) -- example: ['U001', 'U002']
  , amount      integer -- example: [200, 500, 800]
  , stamp       varchar(255) -- example: ['2017-01-30 10:00:00', '2017-02-10 10:00:00', '2017-02-12 10:00:00']
)

CREATE TABLE product_sales (
    category_name varchar(255) -- example: ['dvd', 'cd', 'book']
  , product_id    varchar(255) -- example: ['D001', 'D002', 'D003']
  , sales         integer -- example: [50000, 20000, 10000]
)

