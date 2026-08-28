CREATE TABLE jobs (
    `job_id` integer, -- job id. unique id number identifying the jobs.
    `job_desc` text, -- job description. staff should be mentioned.
    `min_lvl` integer, -- min level. min job level.
    `max_lvl` integer, -- max level. max job level. level range for jobs mentioned in job_desc is (min_lvl, max_lvl).
    primary key (job_id)
);

CREATE TABLE pub_info (
    `pub_id` text, -- publication id. unique id number identifying publications.
    `logo` blob, -- logo of publications.
    `pr_info` text, -- publisher's information.
    primary key (pub_id),
    foreign key (pub_id) references publishers(pub_id)
);

CREATE TABLE sales (
    `stor_id` text, -- store id. id number identifying stores.
    `ord_num` text, -- order number. id number identifying the orders.
    `ord_date` datetime, -- order date. the date of the order.
    `qty` integer, -- quantity. quantity of sales.
    `payterms` text, -- payments.
    `title_id` text, -- title id. id number identifying titles.
    primary key (stor_id, ord_num, title_id),
    foreign key (stor_id)   references stores(stor_id)
    foreign key (title_id)  references titles(title_id)
);

CREATE TABLE publishers (
    `pub_id` text, -- publisher id. unique id number identifying publisher.
    `pub_name` text, -- publisher name.
    `city` text, -- No description.
    `state` text, -- No description.
    `country` text, -- No description.
    primary key (pub_id)
);

CREATE TABLE titleauthor (
    `au_id` text, -- author id.
    `title_id` text, -- title id.
    `au_ord` integer, -- author ordering.
    `royaltyper` integer, -- royaltyper.
    primary key (au_id, title_id),
    foreign key (au_id) references authors(au_id)
    foreign key (title_id)    references titles (title_id)
);

CREATE TABLE employee (
    `emp_id` text, -- employee id. unique number identifying employees.
    `fname` text, -- first name. first name of employees.
    `minit` text, -- middle name.
    `lname` text, -- last name.
    `job_id` integer, -- job id. number identifying jobs.
    `job_lvl` integer, -- job level. higher value means job level is higher.
    `pub_id` text, -- publisher id. id number identifying publishers.
    `hire_date` datetime, -- hire date.
    primary key (emp_id),
    foreign key (job_id) references jobs(job_id)
    foreign key (pub_id) references publishers(pub_id)
);

CREATE TABLE titles (
    `title_id` text, -- title id.
    `title` text, -- title.
    `type` text, -- type of titles.
    `pub_id` text, -- publisher id.
    `price` real, -- price.
    `advance` real, -- pre-paid amount.
    `royalty` integer, -- royalty.
    `ytd_sales` integer, -- year to date sales.
    `notes` text, -- notes if any. had better understand notes contents and put some of them into questions if any.
    `pubdate` datetime, -- publication date.
    primary key (title_id),
    foreign key (pub_id) references publishers(pub_id)
);

CREATE TABLE authors (
    `au_id` text, -- author id. unique number identifying authors.
    `au_lname` text, -- author last name.
    `au_fname` text, -- author first name.
    `phone` text, -- phone number.
    `address` text, -- address.
    `city` text, -- city.
    `state` text, -- state.
    `zip` text, -- zip code.
    `contract` text, -- contract status. 0: not on the contract. 1: on the contract.
    primary key (au_id)
);

CREATE TABLE stores (
    `stor_id` text, -- store id. unique id number of stores.
    `stor_name` text, -- store name.
    `stor_address` text, -- store address.
    `city` text, -- city name.
    `state` text, -- state code.
    `zip` text, -- zip code.
    primary key (stor_id)
);

CREATE TABLE discounts (
    `discounttype` text, -- discount type.
    `stor_id` text, -- store id.
    `lowqty` integer, -- low quantity. low quantity (quantity floor). The minimum quantity to enjoy the discount.
    `highqty` integer, -- high quantity. high quantity (max quantity). The maximum quantity to enjoy the discount.
    `discount` real, -- No description.
    foreign key (stor_id)  references stores(stor_id)
);

CREATE TABLE roysched (
    `title_id` text, -- unique id number identifying title.
    `lorange` integer, -- low range.
    `hirange` integer, -- high range.
    `royalty` integer, -- royalty.
    foreign key (title_id)  references titles(title_id)
);

