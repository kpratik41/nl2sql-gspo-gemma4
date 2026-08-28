CREATE TABLE customer (
    `c_custkey` integer, -- customer key. unique id number identifying the customer.
    `c_mktsegment` text, -- customer market segment. the segment of the customer.
    `c_nationkey` integer, -- customer nation key. the nation key number of the customer.
    `c_name` text, -- customer name. name of the customer.
    `c_address` text, -- customer address. address of the customer.
    `c_phone` text, -- customer phone. phone of the customer.
    `c_acctbal` real, -- customer account balance. account balance. if c_acctbal < 0: this account is in debt.
    `c_comment` text, -- customer comment. comment of the customer.
    PRIMARY KEY (c_custkey),
    FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE nation (
    `n_nationkey` integer, -- nation key. unique id number identifying the nation.
    `n_name` text, -- nation name. name of the nation.
    `n_regionkey` integer, -- nation region key. region key number of the nation.
    `n_comment` text, -- nation comment. comment description of nation.
    primary key (n_nationkey),
    foreign key (n_regionkey) references region (r_regionkey)
);

CREATE TABLE part (
    `p_partkey` integer, -- part key. unique id number identifying the part.
    `p_type` text, -- part type. type of part.
    `p_size` integer, -- part size. size number of part. larger number --> larger part.
    `p_brand` text, -- part brand. brand of part.
    `p_name` text, -- part name. name of part.
    `p_container` text, -- part container. container of part.
    `p_mfgr` text, -- part manufacturer. manufacturer of part.
    `p_retailprice` real, -- part retail price. retail price of part.
    `p_comment` text, -- part comment. comment description text of part.
    primary key (p_partkey)
);

CREATE TABLE lineitem (
    `l_shipdate` date, -- line item ship date. ship date of line item.
    `l_orderkey` integer, -- line item order key. order key number of line item.
    `l_discount` real, -- line item discount. discount of line item. 0.1: 10% off.
    `l_extendedprice` real, -- line item extended price. extended price of line item. full price before discount. discounted price = (l_extendedprice*(1-l_discount)).
    `l_suppkey` integer, -- line item suppkey. suppkey of line item.
    `l_quantity` integer, -- line item quantity. quantity of line item.
    `l_returnflag` text, -- line item return flag. return flag of line item.  R ? returned item.  A and "N": not returned item.
    `l_partkey` integer, -- line item part key. part key of line item.
    `l_linestatus` text, -- line item line status. line status of line item. NOT USEFUL.
    `l_tax` real, -- line item tax. tax of line item. charge = l_extendedprice * (1- l_discount) * (1+ l_tax).
    `l_commitdate` date, -- line item commit date. commit date of line item.
    `l_receiptdate` date, -- line item receipt date. receipt date of line item. freight / shipping / delivery time = l_receiptdate - l_commitdate; less shipping / delivery time --> faster shipping speed.
    `l_shipmode` text, -- line item ship mode. ship mode of line item.
    `l_linenumber` integer, -- line item line number. unique id number identifying the line item.
    `l_shipinstruct` text, -- line item ship instruct. ship instruct of line item.
    `l_comment` text, -- line item comment. comment of line item.
    primary key (l_orderkey, l_linenumber),
    foreign key (l_orderkey) references orders (o_orderkey)
    foreign key (l_partkey, l_suppkey) references partsupp (ps_partkey, ps_suppkey)
);

CREATE TABLE partsupp (
    `ps_partkey` integer, -- part supply part key. unique id number identifying the part key.
    `ps_suppkey` integer, -- part supply supply key. unique id number identifying the supply key.
    `ps_supplycost` real, -- part supply cost. cost number of the part supply. profit = (l_extendedprice*(1-l_discount)) - (ps_supplycost * l_quantity)] l_quantity comes from table lineitem!
    `ps_availqty` integer, -- part supply available quantity. available quantity of the part supply. if ps_availqty < 10 --> closer to OOS (out of stock).
    `ps_comment` text, -- part supply comment. comment description of the part supply.
    primary key (ps_partkey, ps_suppkey),
    foreign key (ps_partkey) references part (p_partkey)
    foreign key (ps_suppkey) references supplier (s_suppkey)
);

CREATE TABLE region (
    `r_regionkey` integer, -- region key. unique id number identifying the region.
    `r_name` text, -- region name. name of the region.
    `r_comment` text, -- region comment. comment description of the region.
    primary key (r_regionkey)
);

CREATE TABLE orders (
    `o_orderdate` date, -- order date. date of the order. earlier order date --> higher priority in delivery.
    `o_orderkey` integer, -- order key. unique id number identifying the order.
    `o_custkey` integer, -- order customer key. customer key of order.
    `o_orderpriority` text, -- order priority. priority rate of order. A smaller number means a higher priority: 0-Urgent.
    `o_shippriority` integer, -- order ship priority. ship priority of order. NOT USEFUL.
    `o_clerk` text, -- order clerk. clerk of order.
    `o_orderstatus` text, -- order status. status of order. NOT USEFUL.
    `o_totalprice` real, -- order total price. total price of the order.
    `o_comment` text, -- order comment. comment description of order.
    primary key (o_orderkey),
    foreign key (o_custkey) references customer (c_custkey)
);

CREATE TABLE supplier (
    `s_suppkey` integer, -- supply key. unique id number identifying the supply.
    `s_nationkey` integer, -- supplier nation key. nation key of the supply.
    `s_comment` text, -- supplier comment. comment description of the comment.
    `s_name` text, -- supplier name. name of the supplier.
    `s_address` text, -- supplier address. address of the supplier.
    `s_phone` text, -- supplier phone. phone number of the supplier.
    `s_acctbal` real, -- supplier account balance. account balance of the supplier. if c_acctbal < 0: this account is in debt.
    primary key (s_suppkey),
    foreign key (s_nationkey) references nation (n_nationkey)
);

