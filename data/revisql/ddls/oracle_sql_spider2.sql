CREATE TABLE customers (
    id INTEGER PRIMARY KEY, -- example: [50042, 50741, 51007]
    name TEXT NOT NULL -- example: ['The White Hart', 'Hygge og Humle', 'Boom Beer Bar']
)

CREATE TABLE conway_gen_zero (
    x INTEGER NOT NULL, -- example: NULL
    y INTEGER NOT NULL, -- example: NULL
    alive INTEGER NOT NULL CHECK (alive IN (0, 1)), -- example: NULL
    PRIMARY KEY (x, y)
)

CREATE TABLE web_devices (
    day  PRIMARY KEY, -- example: ['2019-05-01', '2019-05-02']
    pc INTEGER, -- example: [1042, 967]
    tablet INTEGER, -- example: [812, 1102]
    phone INTEGER -- example: [1610, 2159]
)

CREATE TABLE web_demographics (
    day  PRIMARY KEY, -- example: ['2019-05-01', '2019-05-02']
    m_tw_cnt INTEGER, -- example: [1232, 1438]
    m_tw_qty INTEGER, -- example: [86, 142]
    m_fb_cnt INTEGER, -- example: [1017, 1198]
    m_fb_qty INTEGER, -- example: [64, 70]
    f_tw_cnt INTEGER, -- example: [651, 840]
    f_tw_qty INTEGER, -- example: [76, 92]
    f_fb_cnt INTEGER, -- example: [564, 752]
    f_fb_qty INTEGER -- example: [68, 78]
)

CREATE TABLE channels_dim (
    id INTEGER PRIMARY KEY, -- example: [42, 44]
    name TEXT NOT NULL, -- example: ['Twitter', 'Facebook']
    shortcut TEXT NOT NULL -- example: ['tw', 'fb']
)

CREATE TABLE gender_dim (
    letter TEXT PRIMARY KEY, -- example: ['F', 'M']
    name TEXT -- example: ['Female', 'Male']
)

CREATE TABLE packaging (
    id INTEGER PRIMARY KEY, -- example: [501, 502, 511]
    name TEXT NOT NULL -- example: ['Bottle 330cl', 'Bottle 500cl', 'Gift Carton']
)

CREATE TABLE packaging_relations (
    packaging_id INTEGER NOT NULL, -- example: [511, 521, 522]
    contains_id INTEGER NOT NULL, -- example: [501, 502, 511]
    qty INTEGER NOT NULL, -- example: [3, 2, 72]
    PRIMARY KEY (packaging_id, contains_id),
    FOREIGN KEY (packaging_id) REFERENCES packaging(id),
    FOREIGN KEY (contains_id) REFERENCES packaging(id)
)

CREATE TABLE product_groups (
    id INTEGER PRIMARY KEY, -- example: [142, 152, 202]
    name TEXT NOT NULL -- example: ['Stout', 'Belgian', 'Wheat']
)

CREATE TABLE products (
    id INTEGER PRIMARY KEY, -- example: [4040, 4160, 4280]
    name TEXT NOT NULL, -- example: ['Coalminers Sweat', 'Reindeer Fuel', 'Hoppy Crude Oil']
    group_id INTEGER NOT NULL, -- example: [142, 152, 202]
    FOREIGN KEY (group_id) REFERENCES product_groups(id)
)

CREATE TABLE monthly_sales (
    product_id INTEGER NOT NULL, -- example: [4040, 4160, 4280]
    mth TEXT NOT NULL, -- example: ['2016-01-01', '2016-02-01', '2016-03-01']
    qty INTEGER NOT NULL, -- example: [42, 37, 39]
    PRIMARY KEY (product_id, mth),
    FOREIGN KEY (product_id) REFERENCES products(id),
    CHECK (strftime('%d', mth) = '01')
)

CREATE TABLE breweries (
    id INTEGER PRIMARY KEY, -- example: [518, 523, 536]
    name TEXT NOT NULL -- example: ['Balthazar Brauerei', 'Happy Hoppy Hippo', 'Brewing Barbarian']
)

CREATE TABLE purchases (
    id INTEGER PRIMARY KEY, -- example: [601, 611, 621]
    purchased TEXT NOT NULL, -- example: ['2016-01-01', '2016-01-03', '2016-01-07']
    brewery_id INTEGER NOT NULL, -- example: [518, 523, 536]
    product_id INTEGER NOT NULL, -- example: [4040, 4160, 4280]
    qty INTEGER NOT NULL, -- example: [52, 17, 34]
    cost REAL NOT NULL, -- example: [388.0, 122.0, 163.0]
    FOREIGN KEY (brewery_id) REFERENCES breweries(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
)

CREATE TABLE product_alcohol (
    product_id INTEGER PRIMARY KEY, -- example: [4040, 4160, 4280]
    sales_volume REAL NOT NULL, -- example: [330.0, 500.0]
    abv REAL NOT NULL, -- example: [8.5, 6.0, 7.0]
    FOREIGN KEY (product_id) REFERENCES products(id)
)

CREATE TABLE customer_favorites (
    customer_id INTEGER NOT NULL, -- example: [50042, 50741, 51007]
    favorite_list TEXT, -- example: ['4040,5310', '5430,7790,7870', '6520']
    PRIMARY KEY (customer_id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
)

CREATE TABLE customer_reviews (
    customer_id INTEGER NOT NULL, -- example: [50042, 50741, 51007]
    review_list TEXT, -- example: ['4040:A,6600:C,7950:B', '4160:A', '4280:B,7790:B']
    PRIMARY KEY (customer_id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
)

CREATE TABLE locations (
    id INTEGER PRIMARY KEY, -- example: [1, 2, 3]
    warehouse INTEGER NOT NULL, -- example: [1, 2]
    aisle TEXT NOT NULL, -- example: ['A', 'B', 'C']
    position INTEGER NOT NULL, -- example: [1, 2, 3]
    UNIQUE (warehouse, aisle, position)
)

CREATE TABLE inventory (
    id INTEGER PRIMARY KEY, -- example: [1148, 1151, 1154]
    location_id INTEGER NOT NULL, -- example: [2, 3, 4]
    product_id INTEGER NOT NULL, -- example: [4040, 4160, 4280]
    purchase_id INTEGER NOT NULL, -- example: [719, 720, 721]
    qty REAL NOT NULL, -- example: [11.0, 48.0, 36.0]
    FOREIGN KEY (location_id) REFERENCES locations(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (purchase_id) REFERENCES purchases(id)
)

CREATE TABLE orders (
    id INTEGER PRIMARY KEY, -- example: [421, 427, 429]
    customer_id INTEGER NOT NULL, -- example: [50042, 50741, 51069]
    ordered TEXT, -- example: ['2019-01-15', '2019-01-17', '2019-01-18']
    delivery TEXT, -- example: NULL
    FOREIGN KEY (customer_id) REFERENCES customers(id)
)

CREATE TABLE orderlines (
    id INTEGER PRIMARY KEY, -- example: [9120, 9122, 9233]
    order_id INTEGER NOT NULL, -- example: [421, 422, 423]
    product_id INTEGER NOT NULL, -- example: [4280, 6520, 6600]
    qty REAL NOT NULL, -- example: [110.0, 140.0, 80.0]
    amount REAL NOT NULL, -- example: [2400.0, 2250.0, 1750.0]
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
)

CREATE TABLE monthly_budget (
    product_id INTEGER NOT NULL, -- example: [6520, 6600]
    mth TEXT NOT NULL, -- example: ['2018-01-01', '2018-02-01', '2018-03-01']
    qty REAL NOT NULL, -- example: [30.0, 40.0, 50.0]
    PRIMARY KEY (product_id, mth),
    FOREIGN KEY (product_id) REFERENCES products(id),
    CHECK (strftime('%d', mth) = '01')
)

CREATE TABLE product_minimums (
    product_id INTEGER PRIMARY KEY, -- example: [6520, 6600]
    qty_minimum REAL NOT NULL, -- example: [100.0, 30.0]
    qty_purchase REAL NOT NULL, -- example: [400.0, 100.0]
    FOREIGN KEY (product_id) REFERENCES products(id)
)

CREATE TABLE stock (
    symbol TEXT PRIMARY KEY, -- example: ['BEER']
    company TEXT NOT NULL -- example: ['Good Beer Trading Co']
)

CREATE TABLE ticker (
    symbol TEXT NOT NULL, -- example: ['BEER']
    day TEXT NOT NULL, -- example: ['2019-04-01', '2019-04-02', '2019-04-03']
    price REAL NOT NULL, -- example: [14.9, 14.2, 15.7]
    PRIMARY KEY (symbol, day),
    FOREIGN KEY (symbol) REFERENCES stock(symbol)
)

CREATE TABLE web_apps (
    id INTEGER PRIMARY KEY, -- example: [542]
    name TEXT NOT NULL -- example: ['Webshop']
)

CREATE TABLE web_pages (
    app_id INTEGER NOT NULL, -- example: [542]
    page_no INTEGER NOT NULL, -- example: [1, 2, 3]
    friendly_url TEXT NOT NULL, -- example: ['/Shop', '/Categories', '/Breweries']
    PRIMARY KEY (app_id, page_no),
    FOREIGN KEY (app_id) REFERENCES web_apps(id)
)

CREATE TABLE web_counter_hist (
    app_id INTEGER NOT NULL, -- example: [542]
    page_no INTEGER NOT NULL, -- example: [1, 2, 3]
    day TEXT NOT NULL, -- example: ['2019-04-01', '2019-04-02', '2019-04-03']
    counter INTEGER NOT NULL, -- example: [5010, 5088, 5160]
    PRIMARY KEY (app_id, page_no, day),
    FOREIGN KEY (app_id, page_no) REFERENCES web_pages(app_id, page_no)
)

CREATE TABLE server_heartbeat (
    server TEXT NOT NULL, -- example: ['10.0.0.100', '10.0.0.142']
    beat_time TEXT NOT NULL, -- example: ['2019-04-10 13:00', '2019-04-10 13:05', '2019-04-10 13:10']
    UNIQUE (server, beat_time)
)

CREATE TABLE web_page_visits (
    client_ip TEXT NOT NULL, -- example: ['104.130.89.12', '85.237.86.200']
    visit_time TEXT NOT NULL, -- example: ['2019-04-20 08:15:42', '2019-04-20 08:16:31', '2019-04-20 08:28:55']
    app_id INTEGER NOT NULL, -- example: [542]
    page_no INTEGER NOT NULL, -- example: [1, 2, 3]
    FOREIGN KEY (app_id, page_no) REFERENCES web_pages(app_id, page_no)
)

CREATE TABLE employees (
    id INTEGER PRIMARY KEY, -- example: [142, 144, 147]
    name TEXT NOT NULL, -- example: ['Harold King', 'Mogens Juel', 'Axel de Proef']
    title TEXT NOT NULL, -- example: ['Managing Director', 'IT Manager', 'Product Director']
    supervisor_id INTEGER, -- example: [142, 143, 144]
    FOREIGN KEY (supervisor_id) REFERENCES employees(id)
)

CREATE TABLE emp_hire_periods (
    emp_id INTEGER NOT NULL, -- example: [142, 143, 144]
    start_ TEXT NOT NULL, -- example: ['2010-07-01', '2012-04-01', '2014-01-01']
    end_ TEXT, -- example: ['2012-04-01', '2014-01-01', '2016-06-01']
    title TEXT NOT NULL, -- example: ['Product Director', 'Managing Director', 'IT Technician']
    PRIMARY KEY (emp_id, start_),
    FOREIGN KEY (emp_id) REFERENCES employees(id)
)

CREATE TABLE picking_list (
    id INTEGER PRIMARY KEY, -- example: [841, 842]
    created TEXT NOT NULL, -- example: ['2019-01-16 14:03:41', '2019-01-19 15:57:42']
    picker_emp_id INTEGER, -- example: [149, 152]
    FOREIGN KEY (picker_emp_id) REFERENCES employees(id)
)

CREATE TABLE picking_line (
    picklist_id INTEGER NOT NULL, -- example: [841, 842]
    line_no INTEGER NOT NULL, -- example: [1, 2, 3]
    location_id INTEGER NOT NULL, -- example: [16, 29, 65]
    order_id INTEGER NOT NULL, -- example: [421, 422, 423]
    product_id INTEGER NOT NULL, -- example: [4280, 6520]
    qty REAL NOT NULL, -- example: [42.0, 14.0, 20.0]
    PRIMARY KEY (picklist_id, line_no),
    FOREIGN KEY (picklist_id) REFERENCES picking_list(id),
    FOREIGN KEY (location_id) REFERENCES locations(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
)

CREATE TABLE picking_log (
    picklist_id INTEGER NOT NULL, -- example: [841, 842]
    log_time TEXT NOT NULL, -- example: ['2019-01-16 14:05:11', '2019-01-16 14:05:44', '2019-01-16 14:05:52']
    activity TEXT NOT NULL CHECK (activity IN ('A', 'P', 'D')), -- example: ['D', 'A', 'P']
    location_id INTEGER, -- example: [16, 29, 65]
    pickline_no INTEGER, -- example: [1, 2, 3]
    PRIMARY KEY (picklist_id, log_time),
    FOREIGN KEY (picklist_id) REFERENCES picking_list(id),
    FOREIGN KEY (location_id) REFERENCES locations(id),
    FOREIGN KEY (picklist_id, pickline_no) REFERENCES picking_line(picklist_id, line_no),
    CHECK (NOT (activity = 'P' AND pickline_no IS NULL))
)

CREATE TABLE id_name_type (
    id INTEGER, -- example: NULL
    name TEXT, -- example: NULL
    PRIMARY KEY (id)
)

CREATE TABLE id_name_coll_type (
    collection_id INTEGER PRIMARY KEY -- example: NULL
    -- Additional metadata or constraints if needed
)

CREATE TABLE id_name_coll_entries (
    collection_id INTEGER, -- example: NULL
    id INTEGER, -- example: NULL
    name TEXT, -- example: NULL
    PRIMARY KEY (collection_id, id),  -- Assuming id is unique per collection
    FOREIGN KEY (collection_id) REFERENCES id_name_coll_type(collection_id)
)

CREATE TABLE favorite_coll_type (
    id INTEGER PRIMARY KEY -- example: NULL
)

