CREATE TABLE aircrafts_data (
    aircraft_code character(3) NOT NULL, -- example: ['773', '763', 'SU9']
    model jsonb NOT NULL, -- example: ['{"en": "Boeing 777-300", "ru": "Боинг 777-300"}', '{"en": "Boeing 767-300", "ru": "Боинг 767-300"}', '{"en": "Sukhoi Superjet-100", "ru": "Сухой Суперджет-100"}']
    range integer NOT NULL, -- example: [11100, 7900, 3000]
    CONSTRAINT aircrafts_range_check CHECK ((range > 0))
)

CREATE TABLE airports_data (
    airport_code character(3) NOT NULL, -- example: ['YKS', 'MJZ', 'KHV']
    airport_name jsonb NOT NULL, -- example: ['{"en": "Yakutsk Airport", "ru": "Якутск"}', '{"en": "Mirny Airport", "ru": "Мирный"}', '{"en": "Khabarovsk-Novy Airport", "ru": "Хабаровск-Новый"}']
    city jsonb NOT NULL, -- example: ['{"en": "Yakutsk", "ru": "Якутск"}', '{"en": "Mirnyj", "ru": "Мирный"}', '{"en": "Khabarovsk", "ru": "Хабаровск"}']
    coordinates point NOT NULL, -- example: ['(129.77099609375,62.0932998657226562)', '(114.03900146484375,62.534698486328125)', '(135.18800354004,48.5279998779300001)']
    timezone text NOT NULL -- example: ['Asia/Yakutsk', 'Asia/Vladivostok', 'Asia/Kamchatka']
)

CREATE TABLE boarding_passes (
    ticket_no character(13) NOT NULL, -- example: ['0005435212351', '0005435212386', '0005435212381']
    flight_id integer NOT NULL, -- example: [30625, 24836, 2055]
    boarding_no integer NOT NULL, -- example: [1, 2, 3]
    seat_no character varying(4) NOT NULL -- example: ['2D', '3G', '4H']
)

CREATE TABLE bookings (
    book_ref character(6) NOT NULL, -- example: ['00000F', '000012', '000068']
    book_date timestamp with time zone NOT NULL, -- example: ['2017-07-05 03:12:00+03', '2017-07-14 09:02:00+03', '2017-08-15 14:27:00+03']
    total_amount numeric(10,2) NOT NULL -- example: [265700, 37900, 18100]
)

CREATE TABLE flights (
    flight_id integer NOT NULL, -- example: [1185, 3979, 4739]
    flight_no character(6) NOT NULL, -- example: ['PG0134', 'PG0052', 'PG0561']
    scheduled_departure timestamp with time zone NOT NULL, -- example: ['2017-09-10 09:50:00+03', '2017-08-25 14:50:00+03', '2017-09-05 12:30:00+03']
    scheduled_arrival timestamp with time zone NOT NULL, -- example: ['2017-09-10 14:55:00+03', '2017-08-25 17:35:00+03', '2017-09-05 14:15:00+03']
    departure_airport character(3) NOT NULL, -- example: ['DME', 'VKO', 'SVO']
    arrival_airport character(3) NOT NULL, -- example: ['BTK', 'HMA', 'AER']
    status character varying(20) NOT NULL, -- example: ['Scheduled', 'Cancelled', 'Arrived']
    aircraft_code character(3) NOT NULL, -- example: ['319', 'CR2', '763']
    actual_departure timestamp with time zone, -- example: ['\N', '2017-07-16 09:44:00+03', '2017-08-05 19:06:00+03']
    actual_arrival timestamp with time zone -- example: ['\N', '2017-07-16 10:39:00+03', '2017-08-05 20:01:00+03']
)

CREATE TABLE seats (
    aircraft_code character(3) NOT NULL, -- example: ['319', '320', '321']
    seat_no character varying(4) NOT NULL, -- example: ['2A', '2C', '2D']
    fare_conditions character varying(10) NOT NULL -- example: ['Business', 'Economy', 'Comfort']
)

CREATE TABLE ticket_flights (
    ticket_no character(13) NOT NULL, -- example: ['0005432159776', '0005435212351', '0005435212386']
    flight_id integer NOT NULL, -- example: [30625, 24836, 2055]
    fare_conditions character varying(10) NOT NULL, -- example: ['Business', 'Comfort', 'Economy']
    amount numeric(10,2) NOT NULL -- example: [42100, 23900, 14000]
)

CREATE TABLE tickets (
    ticket_no character(13) NOT NULL, -- example: ['0005432000987', '0005432000988', '0005432000989']
    book_ref character(6) NOT NULL, -- example: ['06B046', 'E170C3', 'F313DD']
    passenger_id character varying(20) NOT NULL) -- example: ['8149 604011', '8499 420203', '1011 752484']

