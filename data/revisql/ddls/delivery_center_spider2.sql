CREATE TABLE channels (
        channel_id INTEGER PRIMARY KEY, -- example: [1, 2, 3]
        channel_name VARCHAR(50), -- example: ['OTHER PLACE', 'PHONE PLACE', 'WHATS PLACE']
        channel_type VARCHAR(50) -- example: ['OWN CHANNEL', 'MARKETPLACE']
    )

CREATE TABLE drivers (
        driver_id INTEGER PRIMARY KEY, -- example: [133, 138, 140]
        driver_modal VARCHAR(50), -- example: ['MOTOBOY', 'BIKER']
        driver_type VARCHAR(50) -- example: ['LOGISTIC OPERATOR', 'FREELANCE']
    )

CREATE TABLE deliveries (
        delivery_id INTEGER PRIMARY KEY, -- example: [2174658, 2174660, 2174661]
        delivery_order_id INTEGER, -- example: [68413340, 68414309, 68416230]
        driver_id INTEGER NULL, -- example: [8378, 2473, 7615]
        delivery_distance_meters DECIMAL(10, 2), -- example: [5199, 410, 3784]
        delivery_status VARCHAR(50) -- example: ['DELIVERED', 'CANCELLED', 'DELIVERING']
    )

CREATE TABLE hubs (
        hub_id INTEGER PRIMARY KEY, -- example: [2, 3, 4]
        hub_name VARCHAR(50), -- example: ['BLUE SHOPPING', 'GREEN SHOPPING', 'RED SHOPPING']
        hub_city VARCHAR(50), -- example: ['PORTO ALEGRE', 'RIO DE JANEIRO', 'SÃO PAULO']
        hub_state CHAR(2), -- example: ['RS', 'RJ', 'SP']
        hub_latitude DECIMAL(9, 6), -- example: [-30.0474148, -30.0374149, -30.0219481]
        hub_longitude DECIMAL(9, 6) -- example: [-51.21351, -51.20352, -51.2083816]
    )

CREATE TABLE payments (
        payment_id INTEGER PRIMARY KEY, -- example: [4427917, 4427918, 4427941]
        payment_order_id INTEGER, -- example: [68410055, 68412721, 68413340]
        payment_amount DECIMAL(10, 2), -- example: [118.44, 394.81, 206.95]
        payment_fee DECIMAL(10, 2), -- example: [0, 7.9, 5.59]
        payment_method VARCHAR(50), -- example: ['VOUCHER', 'ONLINE', 'DEBIT']
        payment_status VARCHAR(50) -- example: ['PAID', 'CHARGEBACK', 'AWAITING']
    )

CREATE TABLE stores (
        store_id INTEGER PRIMARY KEY, -- example: [3, 6, 8]
        hub_id INTEGER, -- example: [2, 3, 8]
        store_name VARCHAR(50), -- example: ['CUMIURI', 'PIMGUCIS DA VIVA ', 'RASMUR S ']
        store_segment VARCHAR(50), -- example: ['FOOD', 'GOOD']
        store_plan_price DECIMAL(10, 2), -- example: [0, 49, 49.9]
        store_latitude DECIMAL(9, 6), -- example: [-30.0374149, -22.921475, -23.0007498]
        store_longitude DECIMAL(9, 6) -- example: [-51.20352, -43.234822, -43.318364]
    )

CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY, -- example: [68405119, 68405123, 68405206]
        store_id INTEGER, -- example: [3512, 3401, 786]
        channel_id INTEGER, -- example: [5, 35, 13]
        payment_order_id INTEGER, -- example: [68405119, 68405123, 68405206]
        delivery_order_id INTEGER, -- example: [68405119, 68405123, 68405206]
        order_status VARCHAR(50), -- example: ['CANCELED', 'FINISHED']
        order_amount DECIMAL(10, 2), -- example: [62.7, 115.5, 55.9]
        order_delivery_fee DECIMAL(10, 2), -- example: [0, 9.9, 0.01]
        order_delivery_cost DECIMAL(10, 2), -- example: [0, 6, 10.93]
        order_created_hour INTEGER, -- example: [0, 1, 2]
        order_created_minute INTEGER, -- example: [1, 4, 13]
        order_created_day INTEGER, -- example: [1, 2, 3]
        order_created_month INTEGER, -- example: [1, 2, 3]
        order_created_year INTEGER, -- example: [2021]
        order_moment_created DATETIME, -- example: ['1/1/2021 12:01:36 AM', '1/1/2021 12:04:26 AM', '1/1/2021 12:13:07 AM']
        order_moment_accepted DATETIME, -- example: ['1/1/2021 1:57:00 AM', '1/1/2021 2:33:00 AM', '1/1/2021 2:12:17 PM']
        order_moment_ready DATETIME, -- example: ['1/2/2021 6:24:06 PM', '1/1/2021 2:38:15 PM', '1/1/2021 2:29:28 PM']
        order_moment_collected DATETIME, -- example: ['1/2/2021 6:30:44 PM', '1/1/2021 2:38:31 PM', '1/1/2021 2:36:30 PM']
        order_moment_in_expedition DATETIME, -- example: ['1/2/2021 6:31:15 PM', '1/1/2021 2:39:05 PM', '1/1/2021 2:39:02 PM']
        order_moment_delivering DATETIME, -- example: ['1/2/2021 6:35:49 PM', '1/1/2021 2:49:18 PM', '1/1/2021 2:42:08 PM']
        order_moment_delivered DATETIME, -- example: ['1/1/2021 4:22:19 PM', '1/1/2021 3:56:45 PM', '1/1/2021 6:01:30 PM']
        order_moment_finished DATETIME, -- example: ['1/2/2021 6:57:34 PM', '1/1/2021 4:12:36 PM', '1/1/2021 3:31:54 PM']
        order_metric_collected_time DECIMAL(10, 2), -- example: [6.63, 0.27, 7.03]
        order_metric_paused_time DECIMAL(10, 2), -- example: [4.55, 10.22, 3.1]
        order_metric_production_time DECIMAL(10, 2), -- example: [2391.25, 26.07, 14.62]
        order_metric_walking_time DECIMAL(10, 2), -- example: [7.17, 0.83, 9.57]
        order_metric_expediton_speed_time DECIMAL(10, 2), -- example: [11.72, 11.05, 12.67]
        order_metric_transit_time DECIMAL(10, 2), -- example: [21.75, 83.3, 49.78]
        order_metric_cycle_time DECIMAL(10, 2) -- example: [2424.72, 120.42, 77.05]
    )

